import type { EngineError } from "./types";

/*
  One transport, two carriers. In the packaged app the request goes down a
  pipe to the sidecar process. In development the browser cannot speak
  stdio, so the same JSON goes to a loopback port that only exists while
  Vite is running. Nothing leaves the machine either way.
*/

type TauriInvoke = (cmd: string, args: Record<string, unknown>) => Promise<unknown>;

interface TauriWindow {
  __TAURI__?: { core?: { invoke: TauriInvoke }; invoke?: TauriInvoke };
}

const DEV_ENDPOINT = "http://127.0.0.1:8756";
let nextId = 1;

function tauriInvoke(): TauriInvoke | null {
  const bridge = (window as unknown as TauriWindow).__TAURI__;
  return bridge?.core?.invoke ?? bridge?.invoke ?? null;
}

export class EngineFailure extends Error {
  action: string;
  trace?: string;

  constructor(error: EngineError) {
    super(error.message);
    this.action = error.action;
    this.trace = error.trace;
  }
}

export async function call<T>(command: string, params: Record<string, unknown> = {}): Promise<T> {
  const message = { id: nextId++, command, params };
  let response: { ok: boolean; data?: T; error?: EngineError };

  const invoke = tauriInvoke();
  if (invoke) {
    response = (await invoke("engine_call", { message })) as typeof response;
  } else {
    let raw: Response;
    try {
      raw = await fetch(DEV_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(message),
      });
    } catch {
      throw new EngineFailure({
        message: "The statistics engine is not running.",
        action: "Start it in a second terminal with npm run engine, then try again.",
      });
    }
    response = await raw.json();
  }

  if (!response.ok || response.data === undefined) {
    throw new EngineFailure(
      response.error ?? {
        message: "The engine returned nothing.",
        action: "Run the analysis again. If it keeps happening, restart the engine.",
      },
    );
  }
  return response.data;
}

export const engine = {
  ping: () => call<{ version: string; boundary: string }>("ping"),
  newProject: (design: string, alpha: number) =>
    call<{ design: string }>("new_project", { design, alpha }),
  open: (path: string, headerRows: number) =>
    call<import("./types").Dataset>("open", { path, header_rows: headerRows }),
  setVariable: (name: string, changes: Record<string, unknown>) =>
    call<{ variable: import("./types").Variable }>("set_variable", { name, ...changes }),
  suggest: (variables: string[]) =>
    call<{ suggestions: import("./types").Suggestion[] }>("suggest", { variables }),
  run: (test: string, variables: string[], alpha: number) =>
    call<import("./types").RunPayload>("run", { test, variables, alpha }),
};
