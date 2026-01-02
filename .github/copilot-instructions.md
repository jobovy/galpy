# Copilot Instructions for galpy

## Repository Overview

**galpy** is a Python package for galactic dynamics that supports orbit integration in various potentials, distribution function evaluation and sampling, and calculation of action-angle coordinates. It's an astropy-affiliated package with full support for astropy's Quantity framework.

- **Repository Size**: ~44MB (7.1MB galpy/, 3.3MB tests/, 18MB doc/)
- **Languages**: Python (142 files) and C (81 files for performance-critical extensions)
- **Python Versions**: Supports 3.10, 3.11, 3.12, 3.13, and 3.14
- **Key Dependencies**: numpy, scipy, matplotlib (required); GSL 1.14+ (optional, for C extensions)
- **Optional Dependencies**: astropy, astroquery, tqdm, numexpr, numba, JAX, pynbody

## Build and Installation

### Prerequisites

**ALWAYS install GSL before building on Linux/macOS** (required for C extensions):
```bash
# Ubuntu/Debian
sudo apt-get install libgsl-dev

# macOS
brew install gsl libomp
```

**Note on macOS**: Set environment variables for OpenMP support:
```bash
export CFLAGS="-I$(brew --prefix)/include -I/usr/local/opt/libomp/include"
export LDFLAGS="-L$(brew --prefix)/lib -L/usr/local/opt/libomp/lib"
```

### Installing Dependencies

**Install core dependencies BEFORE building**:
```bash
pip install --upgrade numpy scipy matplotlib setuptools cython pytest tqdm numexpr
```

Optional dependencies (install based on feature needs):
```bash
# For astropy support (Quantity with units)
pip install astropy pyerfa

# For orbit name queries
pip install astroquery

# For performance (numba)
pip install numba

# For JAX-based features
pip install jax jaxlib

# For snapshot potentials
pip install pynbody h5py pandas pytz wheel
```

### Building the Package

**CRITICAL: Use `--no-build-isolation` flag** to avoid pip timeout issues during dependency resolution:

```bash
# Standard editable install (for development)
python -m pip install --no-build-isolation -ve .

# Build in-place (for running tests without reinstalling)
python setup.py build_ext --inplace
```

**Environment Variables for Compilation**:
- `GALPY_COMPILE_NO_OPENMP=1`: Disable OpenMP support
- `GALPY_COMPILE_COVERAGE=1`: Enable gcov coverage support
- `GALPY_COMPILE_SINGLE_EXT=1`: Compile all C code into single extension (testing only)
- `GALPY_COMPILE_NO_EXT=1`: Skip C extension compilation (testing only)

**Expected Warning**: You will see a warning about "galpy action-angle-torus C library not installed" unless you manually download the Torus code from https://github.com/jobovy/Torus.git (branch: galpy) into `galpy/actionAngle/actionAngleTorus_c_ext/torus`. This is normal and does not prevent package functionality. Installing the Torus code is only necessary when making changes to `actionAngleTorus.py` or any files under `galpy/actionAngle/actionAngleTorus_c_ext/` directory.

### Verifying Installation

```bash
# Test import
python -c "import galpy; print('galpy version:', galpy.__version__)"

# Run quick test
pytest tests/test_import.py -v
```

## Testing

### Running Tests

**Full test suite takes ~50 minutes to complete**. Tests are organized by module:

```bash
# Run all tests (takes ~50 minutes)
pytest -v tests/

# Run specific test file
pytest tests/test_potential.py -v

# Run tests with specific markers
pytest tests/test_orbit.py -k 'test_energy_jacobi_conservation' -v

# Run with coverage
pytest -v tests/ --cov galpy --cov-config .coveragerc --disable-pytest-warnings
```

**Important Test Dependencies**:
- Tests in `test_orbit.py` involving `from_name` require `astroquery`
- Tests in `test_snapshotpotential.py` require `pynbody`
- Tests in `test_dynamfric.py` and `test_FDMdynamfric.py` require `numba`
- Tests in `test_sphericaldf.py` require `JAX`
- Tests in `test_quantity.py` require `astropy`

### Test Parallelization

The CI splits tests into multiple jobs for parallel execution. Reference `.github/workflows/build.yml` for the exact test file groupings if you need to run a subset of tests that matches CI behavior.

## Linting and Code Style

### Pre-commit Hooks

**Install and run pre-commit before committing**:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files  # Run all hooks manually
```

**Configured Hooks** (`.pre-commit-config.yaml`):
- trailing-whitespace, end-of-file-fixer
- pyupgrade (--py38-plus)
- rst-backticks
- pycln (unused imports)
- isort (import sorting, black profile)
- codespell (spell checking)
- ruff-format (code formatting)

## Project Structure

### Repository Root Files
- `setup.py`: Main build configuration (handles C extensions, GSL detection)
- `pyproject.toml`: Minimal build-system requirements, isort config
- `.pre-commit-config.yaml`: Linting/formatting configuration
- `.coveragerc`: Test coverage configuration
- `CONTRIBUTING.md`: Contributor guidelines (includes potential addition guide)
- `README.dev`: Developer notes for adding C potentials

### Main Code Structure

```
galpy/
├── __init__.py
├── actionAngle/          # Action-angle coordinate calculations
│   ├── actionAngle*.py   # Various action-angle classes
│   └── actionAngle*_c_ext/  # C extensions for performance
├── df/                   # Distribution functions
├── orbit/                # Orbit integration and analysis
│   ├── Orbit.py          # Main Orbit class
│   └── orbit_c_ext/      # C extensions for orbit integration
├── potential/            # Gravitational potentials
│   ├── *Potential.py     # Individual potential classes
│   ├── mwpotentials.py   # Milky Way models
│   └── potential_c_ext/  # C extensions for potentials
├── snapshot/             # N-body snapshot support
└── util/                 # Utility functions

tests/                    # Test suite (~87,828 lines)
├── test_*.py             # Test modules (match galpy subpackages)
└── conftest.py           # pytest configuration

doc/                      # Sphinx documentation
├── Makefile              # Builds to ../../galpy-docs/
└── source/               # Documentation source files
```

### Key Configuration Files
- `.coveragerc`: Coverage omit patterns (snapshot, amuse, some util files)
- `.readthedocs.yaml`: Documentation build (Python 3.11, Ubuntu 22.04)
- `.github/workflows/build.yml`: Linux/macOS CI (most comprehensive)
- `.github/workflows/build_windows.yml`: Windows CI
- `.github/workflows/wheels.yml`: Wheel building for releases

## CI/CD and Validation

### GitHub Actions Workflows

**Main Build Workflow** (`.github/workflows/build.yml`):
- Runs on: Ubuntu, macOS (Intel)
- Python versions: 3.9, 3.10, 3.11, 3.12, 3.13
- **Test splitting**: Tests are split into ~15 jobs per Python version for parallel execution
- **OpenMP disabled on Linux**: CI uses `GALPY_COMPILE_NO_OPENMP=1`
- **Coverage enabled on Linux**: Uses `GALPY_COMPILE_COVERAGE=1` with lcov
- **Typical test warnings**: DeprecationWarning and FutureWarning treated as errors
- **astropy warnings**: Also treated as errors when astropy is installed

**Windows Build Workflow** (`.github/workflows/build_windows.yml`):
- Uses micromamba for GSL installation via conda-forge
- Sets GSL environment variables: `INCLUDE`, `LIB`, `LIBPATH`
- Simpler test splits (no pynbody support)

**Pre-commit CI**:
- Runs automatically on PRs
- Auto-fixes issues when possible

### Coverage Requirements

- Target coverage: Very high (see badges in README)
- Coverage reports uploaded to codecov.io
- C code coverage tracked via lcov on Linux

## Common Workflows

### Adding a New Potential

1. Create new potential class in `galpy/potential/NewPotential.py`
2. Follow existing potential structure (inherit from appropriate base class)
3. Implement required methods: `_evaluate`, `_Rforce`, `_zforce`, etc.
4. Add tests to `tests/test_potential.py`
5. Add API documentation page in `doc/source/reference/potentialnew.rst` (2-line file)
6. **Optional**: Add C implementation following `README.dev` steps
7. Run tests: `pytest tests/test_potential.py -v -k NewPotential` and `pytest tests/test_orbit.py -v -k NewPotential`
8. See PR #760 as a comprehensive example of adding a potential

### Making Code Changes

1. **Install in editable mode**: `python -m pip install --no-build-isolation -ve .`
2. **Make minimal changes**: Surgical edits to specific functions/methods
3. **Build C extensions if changed**: `python setup.py build_ext --inplace`
4. **Run relevant tests**: `pytest tests/test_<module>.py -v`
5. **Run linters**: `pre-commit run --all-files`
6. **Build docs if changed**: `cd doc && make html` (output: `../../galpy-docs/`)

### Debugging Build Issues

**GSL not found**:
```bash
# Check GSL version
gsl-config --version  # Should be ≥1.14 for action-angle-torus

# Set paths manually if needed
export CFLAGS="$CFLAGS -I/path/to/gsl/include/"
export LDFLAGS="$LDFLAGS -L/path/to/gsl/lib/"
```

**C extension compilation errors**:
- Ensure numpy/cython are installed BEFORE building
- Use `--no-build-isolation` to avoid version conflicts
- Check compiler is available (gcc/clang on Linux/macOS, MSVC on Windows)

**Import errors after installation**:
- Check for conflicting `galpy` directory in working directory
- Reinstall with: `pip uninstall galpy && python -m pip install --no-build-isolation -ve .`

## Important Notes

### Memory and Performance
- The test suite is comprehensive and memory-intensive
- Some tests (e.g., `test_streamgapdf.py`, `test_diskdf.py`) are particularly long-running
- CI uses test splitting to parallelize execution
- Use `-k ...` to select relevant tests when working on specific features

### Documentation Changes
- Documentation builds to `../../galpy-docs/` (outside repository)
- Requires sphinx and documentation dependencies

### Python/C Integration
- C extensions significantly speed up orbit integration and potential evaluation
- Most functionality works without C extensions (Python fallback available)
- Adding C implementations requires changes to multiple files (see `README.dev`)

### Deprecation Warnings
- CI treats deprecation warnings as errors
- Fix any new deprecation warnings in your code
- Test with: `pytest -W error::DeprecationWarning`

## Trust These Instructions

These instructions are comprehensive and validated through actual testing. **When implementing changes, trust these instructions and only perform additional searches if information is incomplete or appears incorrect**. The build commands, test procedures, and environment setup documented here have been verified to work correctly.
