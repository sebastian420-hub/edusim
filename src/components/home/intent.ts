/**
 * Rules for when a home-page plate may run its live preview. Pure functions plus a one-at-a-time
 * registry, so the behaviour is unit-testable without a browser.
 */

export interface IntentEnvironment {
  reducedMotion: boolean;
  hasWebGPU: boolean;
  pointerType: string;
}

/** Previews start only for a mouse/pen/keyboard user who allows motion and has WebGPU — never on touch. */
export function mayGoLive({ reducedMotion, hasWebGPU, pointerType }: IntentEnvironment): boolean {
  if (reducedMotion || !hasWebGPU) return false;
  return pointerType === "mouse" || pointerType === "pen" || pointerType === "keyboard";
}

type Deactivate = () => void;

/** At most one plate is live at a time: activating one stops the previous. */
export class SingleLive {
  private current: { id: string; stop: Deactivate } | null = null;

  activate(id: string, stop: Deactivate): void {
    if (this.current?.id === id) {
      this.current = { id, stop };
      return;
    }
    this.current?.stop();
    this.current = { id, stop };
  }

  /** Stops `id` if (and only if) it is the live one. */
  release(id: string): void {
    if (this.current?.id !== id) return;
    this.current.stop();
    this.current = null;
  }

  get activeId(): string | null {
    return this.current?.id ?? null;
  }
}

export const liveRegistry = new SingleLive();
