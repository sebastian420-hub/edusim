"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { ComponentType, ReactNode } from "react";
import { hasWebGPU } from "@/lib/gpu/support";
import { liveRegistry, mayGoLive } from "./intent";

type PreviewComponent = ComponentType<{ id: string; onFail: () => void }>;

// The GPU code (vgpu + the simulations) is only downloaded when a visitor shows intent. A plain import()
// in the handler keeps it out of the page's chunk list; next/dynamic would have it preloaded on load.
const loadPreview = () => import("./LivePreview").then((m) => m.default as PreviewComponent);

/** Time the pointer must stay away before the preview stops, so brushing past an edge doesn't flicker. */
const RELEASE_DELAY_MS = 150;

/**
 * The art area of a plate. Shows the server-rendered poster (`children`); when the visitor hovers or
 * keyboard-focuses the plate, the real simulation runs on top of it. Rules (see intent.ts): one live
 * plate at a time, never on touch, with reduced motion or without WebGPU; paused when scrolled out of
 * view or the tab is hidden; a failure leaves the poster in place for good.
 */
export function PlateArt({ id, children }: { id: string; children: ReactNode }) {
  const root = useRef<HTMLDivElement>(null);
  const [live, setLive] = useState(false);
  const [Preview, setPreview] = useState<PreviewComponent | null>(null);
  const failed = useRef(false);

  const stop = useCallback(() => setLive(false), []);
  const fail = useCallback(() => {
    failed.current = true;
    setLive(false);
  }, []);

  useEffect(() => {
    const plate = root.current?.closest<HTMLElement>("[data-plate]");
    if (!plate) return;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const request = (pointerType: string) => {
      clearTimeout(timer);
      if (failed.current) return;
      const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      if (!mayGoLive({ reducedMotion, hasWebGPU: hasWebGPU(), pointerType })) return;
      liveRegistry.activate(id, stop);
      setLive(true);
      void loadPreview().then((component) => setPreview(() => component));
    };
    const release = () => {
      clearTimeout(timer);
      timer = setTimeout(() => liveRegistry.release(id), RELEASE_DELAY_MS);
    };

    const onEnter = (e: PointerEvent) => request(e.pointerType);
    const onFocusIn = (e: FocusEvent) => {
      if ((e.target as HTMLElement).matches(":focus-visible")) request("keyboard");
    };
    const onVisibility = () => {
      if (document.hidden) liveRegistry.release(id);
    };
    // Scrolled mostly out of view: stop; the next hover starts it again.
    const observer = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting) liveRegistry.release(id);
    }, { threshold: 0.4 });
    observer.observe(plate);

    plate.addEventListener("pointerenter", onEnter);
    plate.addEventListener("pointerleave", release);
    plate.addEventListener("focusin", onFocusIn);
    plate.addEventListener("focusout", release);
    document.addEventListener("visibilitychange", onVisibility);

    // The visitor may already be hovering or have tabbed to the plate before hydration attached the listeners.
    const active = document.activeElement as HTMLElement | null;
    if (plate.contains(active) && active?.matches(":focus-visible")) request("keyboard");
    else if (plate.matches(":hover") && window.matchMedia("(hover: hover)").matches) request("mouse");

    return () => {
      clearTimeout(timer);
      observer.disconnect();
      plate.removeEventListener("pointerenter", onEnter);
      plate.removeEventListener("pointerleave", release);
      plate.removeEventListener("focusin", onFocusIn);
      plate.removeEventListener("focusout", release);
      document.removeEventListener("visibilitychange", onVisibility);
      liveRegistry.release(id);
    };
  }, [id, stop]);

  return (
    <div ref={root} className="absolute inset-0">
      {children}
      {live && Preview && <Preview id={id} onFail={fail} />}
    </div>
  );
}
