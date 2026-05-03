# Bayes Factor

A `BayesFactor` class for binomial data, comparing a slab prior on θ against a narrow spike prior centered at 0.5. Includes a `unittest` test suite.

## Files

- `bayes_factor.py` — the `BayesFactor` class
- `tests/test_bayes_factor.py` — test suite
- `Dockerfile` — builds an image with all dependencies

## Run the tests locally

Requires Python 3 and `scipy`. From this directory:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Run the tests in Docker

From this directory:

```bash
docker build -t bayes-factor .
docker run --rm bayes-factor
```