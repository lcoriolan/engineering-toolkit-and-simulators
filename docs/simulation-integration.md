# Execution and browser control

The shared dispatcher invokes registered implementations, returns their results and preserves exception types. A bound wrapper retains its callable revision after reload. Unknown operations and unavailable engines fail explicitly. Nested calls carry a parent execution identity. Execution records contain control/status metadata rather than argument or result payloads.

Browser initialization and every step chunk require a central grant. The adapter reports diagnostics, consumed steps and typed errors. Paused or stopped workers cannot obtain usable grants, and a lost connection stops further work.

The D2Q9 and D3Q19 demonstration solvers each initialized and advanced through 16 finite steps in the browser checks. Pause held the 2D step count and stop rejected another 3D advance. These six checks establish control behavior and finite state, not flow accuracy or mesh convergence. See the [browser record](../evidence/public-browser-validation.json) and [validation instructions](validation.md).
