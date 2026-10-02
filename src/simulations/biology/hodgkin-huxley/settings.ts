import { boolField, numberField } from "@/lib/urlState";
import type { FieldCodec, Schema } from "@/lib/urlState";
import { DEFAULT_PARAMS, spikesPerCycle, steadyFiringRate } from "./hh";
import type { FiAnalysis, HHParams } from "./hh";

/** The user-adjustable settings: what the sliders control, and what goes into a shared link. */
export interface HHSettings {
  /** Injected current, µA/cm². */
  I_inj: number;
  /** Temperature, °C. */
  temperature: number;
  /** Maximum sodium conductance (lower = more TTX), mS/cm². */
  g_Na: number;
  /** Maximum potassium conductance (lower = more TEA), mS/cm². */
  g_K: number;
  /** Milliseconds of neuron time simulated per real second. */
  timeScale: number;
  pulse_mode: 0 | 1 | 2;
  /** Show the firing-rate curve in the sidebar. */
  showCurve: boolean;
}

export const HH_DEFAULTS: HHSettings = {
  I_inj: DEFAULT_PARAMS.I_inj,
  temperature: DEFAULT_PARAMS.temperature,
  g_Na: DEFAULT_PARAMS.g_Na,
  g_K: DEFAULT_PARAMS.g_K,
  timeScale: 30,
  pulse_mode: 0,
  showCurve: true,
};

const stimulusField: FieldCodec<0 | 1 | 2> = {
  decode: (raw) => (raw === "0" ? 0 : raw === "1" ? 1 : raw === "2" ? 2 : undefined),
  encode: (value) => String(value),
};

export const HH_SCHEMA: Schema<HHSettings> = {
  I_inj: numberField(-20, 50),
  temperature: numberField(0, 40),
  g_Na: numberField(0, 200),
  g_K: numberField(0, 80),
  timeScale: numberField(5, 200),
  pulse_mode: stimulusField,
  showCurve: boolField,
};

/** Full model parameters for a settings object (leak conductance, reversal potentials, capacitance fixed). */
export function toParams(s: Pick<HHSettings, "I_inj" | "temperature" | "g_Na" | "g_K" | "pulse_mode">): HHParams {
  return { ...DEFAULT_PARAMS, I_inj: s.I_inj, temperature: s.temperature, g_Na: s.g_Na, g_K: s.g_K, pulse_mode: s.pulse_mode };
}

/** What the readouts and challenge goals can observe. */
export interface HHReadouts {
  /** Steady firing rate (Hz) under DC current; 0 when silent or in pulse modes. */
  rate: number;
  /** Spikes per 25 ms stimulus cycle in pulse / twin modes (0 in DC mode). */
  spikesPerCycle: number;
  fi: FiAnalysis;
}

export function computeReadouts(settings: HHSettings, fi: FiAnalysis): HHReadouts {
  const params = toParams(settings);
  return settings.pulse_mode === 0
    ? { rate: steadyFiringRate(params), spikesPerCycle: 0, fi }
    : { rate: 0, spikesPerCycle: spikesPerCycle(params), fi };
}
