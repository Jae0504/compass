# MSVBASE Submodule Setup and Benchmark Restore

Use these commands from repo root (`/home/jykang5/compass`).

## 1) Register/init MSVBASE submodule and restore custom benchmark files

```bash
./scripts/setup_msvbase_submodule.sh
```

This does:

- ensures `MSVBASE` is configured as a git submodule,
- runs `git submodule update --init --recursive MSVBASE`,
- restores custom files into `MSVBASE/scripts/`:
  - `sptag_fvec_filter_bench.py`
  - `SPTAG_FVEC_FILTER_BENCH.md`

## 2) Restore benchmark files only (after submodule update/reset)

```bash
./scripts/restore_msvbase_overrides.sh
```

You can also pass an explicit MSVBASE path:

```bash
./scripts/restore_msvbase_overrides.sh /path/to/repo/MSVBASE
```

## 3) Run benchmark script via root wrapper

```bash
./scripts/run_msvbase_sptag_bench.sh --mode prepare --nfilters 100
./scripts/run_msvbase_sptag_bench.sh --mode query --k 10 --num-queries 100 --nfilters 100
```

Or run full pipeline:

```bash
./scripts/run_msvbase_sptag_bench.sh --mode all --k 10 --num-queries 100 --nfilters 100
```

## 4) Override submodule URL at setup time (optional)

```bash
MSVBASE_URL=<your-fork-url> ./scripts/setup_msvbase_submodule.sh
```

## 5) Source of restored files

The canonical copies live in:

- `scripts/msvbase_overrides/sptag_fvec_filter_bench.py`
- `scripts/msvbase_overrides/SPTAG_FVEC_FILTER_BENCH.md`

If you edit benchmark behavior/docs, update these override files.
