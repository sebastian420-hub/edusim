import { init, surface, effect, compute, storage, frameLoop, pingPongStorage } from 'vgpu';
import { shaderSource as computeWgsl } from './compute.wgsl';
import { shaderSource as renderWgsl } from './render.wgsl';
import { patterns, getPatternBounds } from './patterns';

export class CellularAutomataSim {
  private gpu: any = null;
  private canvas: HTMLCanvasElement;
  private targetSurface: any = null;
  private width: number;
  private height: number;
  
  private ppBuffer: any;
  
  private computePass: any;
  private renderPass: any;
  private loopHandle: any = null;
  
  private ruleBirth: number = 0b000001000; // B3
  private ruleSurvive: number = 0b000001100; // S23
  
  private generation: number = 0;
  private theme: number = 0;
  
  private panX: number = 0;
  private panY: number = 0;
  private zoom: number = 1.0;

  private isPlaying: boolean = false;
  private stepsPerSecond: number = 10;
  private lastStepTime: number = 0;

  private hostData: Uint32Array;
  
  public onGenerationChange?: (gen: number) => void;

  constructor(canvas: HTMLCanvasElement, width: number = 256, height: number = 256) {
    this.canvas = canvas;
    this.width = width;
    this.height = height;
    this.hostData = new Uint32Array(width * height);
  }

  public async initialize() {
    this.gpu = await init();
    this.gpu.onError((err: any) => {
      console.error(">>> CS GPU ERROR:", err.message, err.cause, err.detail, err);
    });

    this.targetSurface = surface(this.gpu, this.canvas, { dpr: window.devicePixelRatio });
    
    const bufferSize = this.width * this.height * 4;
    this.ppBuffer = pingPongStorage(this.gpu, bufferSize);

    this.computePass = compute(this.gpu, computeWgsl);
    this.renderPass = effect(this.gpu, renderWgsl);

    this.loadPattern("Glider Gun (Gosper)");

    this.loopHandle = frameLoop(this.gpu, (f) => {
      const now = performance.now();
      const interval = 1000 / this.stepsPerSecond;

      if (this.isPlaying && (now - this.lastStepTime > interval)) {
        this.lastStepTime = now;
        this.runComputeStep();
      }

      this.renderPass.set({
        cells: this.ppBuffer.read,
        params: {
          gridWidth: this.width,
          gridHeight: this.height,
          canvasWidth: this.canvas.width,
          canvasHeight: this.canvas.height,
          panX: this.panX,
          panY: this.panY,
          zoom: this.zoom,
          theme: this.theme
        }
      });

      f.pass(this.targetSurface, this.renderPass);
    });
  }

  public destroy() {
    this.isPlaying = false;
    if (this.loopHandle) {
      this.loopHandle.stop();
    }
    if (this.gpu) {
      this.gpu.dispose();
    }
  }

  private parseRule(ruleStr: string): number {
    let mask = 0;
    for (let i = 0; i < ruleStr.length; i++) {
      const d = parseInt(ruleStr[i]);
      if (!isNaN(d) && d >= 0 && d <= 8) {
        mask |= (1 << d);
      }
    }
    return mask;
  }

  public setRules(birth: string, survive: string) {
    this.ruleBirth = this.parseRule(birth);
    this.ruleSurvive = this.parseRule(survive);
  }

  public setTheme(theme: number) {
    this.theme = theme;
  }
  
  public setSpeed(speed: number) {
    this.stepsPerSecond = speed;
  }

  public setGridSize(size: number) {
    this.width = size;
    this.height = size;
    const bufferSize = this.width * this.height * 4;
    
    if (this.gpu) {
      this.ppBuffer = pingPongStorage(this.gpu, bufferSize);
    }
    
    this.hostData = new Uint32Array(this.width * this.height);
    this.clear();
  }

  public clear() {
    this.hostData.fill(0);
    if (this.ppBuffer) this.ppBuffer.read.write(this.hostData);
    this.generation = 0;
    if (this.onGenerationChange) this.onGenerationChange(0);
  }

  public randomize() {
    for (let i = 0; i < this.hostData.length; i++) {
      this.hostData[i] = Math.random() > 0.8 ? 1 : 0;
    }
    if (this.ppBuffer) this.ppBuffer.read.write(this.hostData);
    this.generation = 0;
    if (this.onGenerationChange) this.onGenerationChange(0);
  }

  public loadPattern(name: string) {
    const pattern = patterns.find(p => p.name === name);
    if (!pattern) return;

    this.hostData.fill(0);
    
    const bounds = getPatternBounds(pattern);
    const startX = Math.floor(this.width / 2 - bounds.width / 2);
    const startY = Math.floor(this.height / 2 - bounds.height / 2);

    for (const [px, py] of pattern.points) {
      const x = startX + px;
      const y = startY + py;
      if (x >= 0 && x < this.width && y >= 0 && y < this.height) {
        this.hostData[y * this.width + x] = 1;
      }
    }
    if (this.ppBuffer) this.ppBuffer.read.write(this.hostData);
    this.generation = 0;
    if (this.onGenerationChange) this.onGenerationChange(0);
  }

  public toggleCell(canvasX: number, canvasY: number) {
    const rect = this.canvas.getBoundingClientRect();
    const u = canvasX / rect.width;
    const v = canvasY / rect.height;

    const gridX_f = (u - this.panX) * (this.width / this.zoom);
    const gridY_f = (v - this.panY) * (this.height / this.zoom);

    const x = Math.floor(gridX_f);
    const y = Math.floor(gridY_f);

    if (x >= 0 && x < this.width && y >= 0 && y < this.height) {
      const offset = (y * this.width + x) * 4;
      const val = new Uint32Array([1]);
      if (this.ppBuffer) this.ppBuffer.read.write(val, offset);
    }
  }

  public play() {
    this.isPlaying = true;
  }

  public pause() {
    this.isPlaying = false;
  }

  public step() {
    this.runComputeStep();
  }

  public setPanZoom(panX: number, panY: number, zoom: number) {
    this.panX = panX;
    this.panY = panY;
    this.zoom = zoom;
  }

  private runComputeStep() {
    if (!this.gpu || !this.computePass) return;

    this.computePass.set({
      cellsIn: this.ppBuffer.read,
      cellsOut: this.ppBuffer.write,
      params: {
        width: this.width,
        height: this.height,
        rule_birth: this.ruleBirth,
        rule_survive: this.ruleSurvive
      }
    });

    const workgroupsX = Math.ceil(this.width / 16);
    const workgroupsY = Math.ceil(this.height / 16);
    this.computePass.dispatch(workgroupsX, workgroupsY, 1);

    this.ppBuffer.swap();

    this.generation++;
    if (this.onGenerationChange) this.onGenerationChange(this.generation);
  }
}
