# MoRa Stats

A desktop statistics application that runs the analysis and then tells you
what the result actually means. The bar for the base app is jamovi. The part
that is new is the interpretation engine, which is deterministic Python:
rule-based, no model, no network, no API key, and auditable line by line.

Data never leaves the machine. The packaged app opens no ports at all.

## Where the project stands

Milestone 1 is partly built, in the order the brief asked for.

| Piece | State |
| --- | --- |
| Interpretation engine, all three blocks and all rules | built, 74 tests passing |
| Descriptives, t-test, Welch, Mann-Whitney, one-way ANOVA, Kruskal-Wallis, correlation, chi-square | built |
| Assumption checking and test suggestion | built |
| .csv and .xlsx import with type detection | built, including multi-row headers |
| Sidecar protocol over stdin and stdout | built and tested |
| Three-pane interface with live results and one chart per analysis | built, runs in the browser against the dev bridge |
| Results as journal tables, copyable into Word and Excel, chart saved as SVG | built |
| Charts with switchable views, offered only where they add something | built |
| Tauri shell that owns the window and the sidecar | written, not yet compiled here, needs Rust on your machine |
| Windows and macOS installer pipeline | written and half proven: the frozen engine runs, the installer step needs Windows or CI |
| Project file, agent layer, .sav, regression | not started, as agreed |

## Running the engine on its own

The engine is a normal Python package with no reference to any user
interface. This is the piece to read first.

```bash
cd engine
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest -q                    # 74 tests
python demo/run_demo.py      # the three blocks on seven known datasets
```

The demo includes Student's sleep data, where the engine reproduces the
published values (t = -1.8608, Welch df = 17.776, p = .0794), a null result
that gets a minimum detectable effect rather than a shrug, and a significant
result from a small design that gets an exaggeration factor.

Single analyses from the command line:

```bash
python -m mora_engine profile ../app/sample/teaching-trial.csv
python -m mora_engine suggest ../app/sample/teaching-trial.csv --vars post_score arm
python -m mora_engine run ../app/sample/teaching-trial.csv \
    --test welch_t --vars post_score arm --design experimental --glossary
```

`--glossary` adds a short explanation of p values, intervals, effect sizes
and power underneath the result, which is the mode to show a student.

## Running the interface

Two terminals, because in development the browser cannot talk to a process
over stdin and stdout.

```bash
# terminal 1, the engine
cd app
npm run engine        # loopback only, development only, never in the build

# terminal 2, the interface
cd app
npm install
npm run dev           # http://localhost:5173
```

The file field at the top is prefilled with `sample/teaching-trial.csv`, a
made-up teaching trial with 72 students, two missing scores and a seven
point confidence scale. Press Open.

### What to check

1. The variable list types `confidence` as ordinal and `faculty` as nominal,
   and each type can be corrected in place from the list.
2. Click `post_score` then `arm`. The test list narrows to the ones that fit
   and greys out ANOVA with the reason attached rather than hiding it.
3. The result appears with no apply step. Change the threshold and it
   recomputes.
4. That comparison is not significant, and the interpretation says why that
   is not the same as no effect, naming the smallest effect the design could
   have caught.
5. Click `study_hours` then `post_score` for the correlation. The language
   switches to association if the project design is set to correlational.
6. Run a third test and the header starts reporting the running count and
   the family-wise false positive rate.
7. Hover a table and use Copy table. It pastes into Word as a real table and
   into Excel as real cells, because the clipboard carries tab separated
   text rather than a picture. Markdown copies the same table for anyone
   drafting elsewhere.
8. Save as SVG under the chart writes a file with the theme colours resolved,
   so it opens correctly outside the app and scales to any print size.
9. Switch the chart between means and intervals, spread, and every case.
   The same observations are redescribed rather than replaced, which is why
   the transition animates. Hover or tab to any group and its figures appear
   in the line underneath.

Charts follow the data rather than the test. A rank test opens on the spread
view, groups with fewer than eight cases skip the box entirely and lead with
the raw points, because a box promises a shape that eight numbers cannot
support. Descriptives get no chart at all and say so. Jitter is seeded in the
engine, so a strip of points never moves between runs or between machines.
Datasets above about twelve hundred cases per group are thinned for drawing
only, and every figure still uses all of them.

Every figure in those tables is formatted by the engine, not by the
interface. A p value below .001 reads as `< .001`, effect sizes bounded by
one lose their leading zero, and intervals keep two decimals, because those
are statistical conventions rather than display choices and they should not
drift between the app and the command line.

## Making an installer people can double click

The end product is one file. On Windows it is a setup `.exe` that installs
for the current user with no administrator prompt. Nobody installs Python,
nobody creates a virtual environment, nobody sees a terminal. The Python
engine is frozen into a single binary that the desktop shell spawns behind
the window.

### Two ways to get the .exe

**Through GitHub, with no Windows machine.** Push the repository, then run
the Build installers workflow, or push a tag such as `v0.1.0`. It builds on
a Windows runner and a macOS runner, and the files appear under Artifacts
when the run finishes. The workflow runs the engine test suite first and
stops if anything fails, because nothing should be packaged from a red
build.

**On a Windows machine.** Install Python 3.11 or newer, Node 20 or newer,
Rust, and the Microsoft C++ build tools, then:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build-windows.ps1
```

The installer lands in `app\src-tauri\target\release\bundle\nsis`.

### What actually happens in that build

1. The engine test suite runs. A red suite stops the build.
2. PyInstaller freezes the engine to `mora-engine.exe` using
   `engine/packaging/mora-engine.spec`. This step is verified: the frozen
   binary was built and driven through the full protocol, opening a file and
   returning a complete result with no Python installed on the path.
3. The binary is copied to `app/src-tauri/binaries/` under the name Tauri
   expects, which carries the target triple:
   `mora-engine-x86_64-pc-windows-msvc.exe`. Getting that name wrong is the
   usual reason a Tauri sidecar is reported as missing at runtime.
4. The frontend is built, and Tauri wraps the window, the web assets and the
   engine binary into one installer.

The frozen engine is around 110 MB, almost all of it SciPy and pandas, which
puts the finished installer near 120 MB. That is the price of shipping real
numerical libraries rather than reimplementing them, and it is a one time
download rather than something the user manages.

UPX compression is switched off in the spec on purpose. It saves perhaps
30 MB and is one of the most reliable ways to have a new unsigned binary
flagged by antivirus software.

### Before the first real release

The icon files listed in `tauri.conf.json` do not exist yet. Generate them
from a single square PNG once the app has a mark:

```bash
cd app && npx --yes @tauri-apps/cli@^2 icon path/to/icon.png
```

## Unsigned builds, and what users will see

Builds are unsigned, so both operating systems will warn about them. This is
worth being honest about in the download page as well as in the app, because
a user who is not warned in advance reads the warning as a sign that
something is wrong.

On macOS the first open shows a message that the app cannot be verified.
Right click the app in Finder, choose Open, then confirm. If the app is
blocked entirely, remove the quarantine flag:

```bash
xattr -dr com.apple.quarantine "/Applications/MoRa Stats.app"
```

On Windows, SmartScreen shows a blue panel saying the publisher is unknown.
Choose More info, then Run anyway. The warning fades as more people install
the same file, since SmartScreen builds reputation per binary, so it will be
loudest for your first few releases and for every new version.

If you later decide to pay for it, an OV code signing certificate removes
the macOS warning through notarisation and quietens Windows over time. An EV
certificate removes the Windows warning immediately. Neither changes a line
of this code: both are a signing step added to the workflow.

The application repeats both of these on first run, since a warning nobody
can act on is not a warning.

## Layout

```
engine/                 the product, usable and testable without any UI
  src/mora_engine/
    interpret.py        results object in, interpretation object out
    power.py            minimum detectable effect and the Type M ratio
    diagnostics.py      the threat checks and their thresholds
    effects.py          effect sizes and their intervals
    session.py          the multiple comparisons ledger
    suggest.py          which tests fit, and why the others do not
    sidecar.py          one JSON object per line, over stdio
    devserver.py        loopback bridge, development only
  tests/                74 tests
  demo/run_demo.py      the engine on known datasets

app/                    React and TypeScript frontend
  src/panes/            data, analysis, results
  src/components/       the effect interval graphic and the charts
  src-tauri/            the Rust shell that owns the window and the sidecar
  sample/               a dataset to open on first run
```

## Conventions the engine uses

Hedges' g with noncentral t intervals. Omega squared for ANOVA. Rank-biserial
and epsilon squared with a seeded bootstrap. Fisher z for correlations, with
the Bonett-Wright correction for Spearman. Cramer's V from a noncentral
chi-square interval.

Shapiro-Wilk is computed and shown but never decides which test is
suggested, because it is underpowered exactly where non-normality matters
and certain to reject exactly where it does not.

Floor and ceiling checks need declared scale limits. When they are missing
the engine says so instead of guessing from the observed range.

The inflation warning on an underpowered significant result is a Type M
exaggeration ratio, not post hoc power, which is a one-to-one function of p
and therefore says nothing new.

The test counter keys on the question rather than the click, so Student,
Welch and Mann-Whitney on the same variables count once.
