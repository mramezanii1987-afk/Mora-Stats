// MoRa Stats desktop shell.
//
// The only job of this process is to own the window and to keep one Python
// sidecar alive behind it. Requests go down the pipe as a single line of
// JSON and come back the same way. No port is opened, so nothing else on the
// machine and nothing on the network can reach the engine.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

use serde_json::Value;
use tauri::Manager;

struct Engine {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Engine {
    fn start() -> Result<Self, String> {
        let mut command = Command::new(sidecar_path());
        command.stdin(Stdio::piped()).stdout(Stdio::piped());
        let mut child = command
            .spawn()
            .map_err(|e| format!("The statistics engine did not start: {e}"))?;
        let stdin = child.stdin.take().ok_or("The engine has no input pipe.")?;
        let stdout = child.stdout.take().ok_or("The engine has no output pipe.")?;
        Ok(Engine { child, stdin, stdout: BufReader::new(stdout) })
    }

    fn call(&mut self, message: &Value) -> Result<Value, String> {
        writeln!(self.stdin, "{message}").map_err(|e| e.to_string())?;
        self.stdin.flush().map_err(|e| e.to_string())?;
        let mut line = String::new();
        self.stdout.read_line(&mut line).map_err(|e| e.to_string())?;
        serde_json::from_str(&line).map_err(|e| e.to_string())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

#[cfg(debug_assertions)]
fn sidecar_path() -> String {
    // During development the engine runs from the source tree.
    std::env::var("MORA_ENGINE").unwrap_or_else(|_| "mora-engine".into())
}

#[cfg(not(debug_assertions))]
fn sidecar_path() -> String {
    "mora-engine".into()
}

#[tauri::command]
fn engine_call(state: tauri::State<Mutex<Engine>>, message: Value) -> Result<Value, String> {
    let mut engine = state.lock().map_err(|_| "The engine is busy.".to_string())?;
    engine.call(&message)
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let engine = Engine::start()?;
            app.manage(Mutex::new(engine));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![engine_call])
        .run(tauri::generate_context!())
        .expect("MoRa Stats failed to start");
}
