# Two-Arm Spiral Point Classifier

This project connects an Anvil web interface to a TensorFlow model that classifies
2D points as belonging to one of two noisy spiral arms. Upload a UTF-8 CSV with
exactly two numeric columns, `x` and `y`; the server returns the predicted arm,
a visualization, and a downloadable CSV. `xy_locs.csv` is a small example input.
Uploads are limited to 10,000 points and non-finite or malformed coordinates are
rejected before model inference.

## Run the server

Install the dependencies with `python -m pip install -r requirements.txt`, then
set the Anvil Uplink key in `ANVIL_UPLINK_KEY` before running the server-side
notebook. Keep the key out of notebooks, source control, and logs. A key was
previously embedded in this notebook's history; rotate it in Anvil before using
the integration.

The client application remains hosted by Anvil. Its source is available through
the [Anvil clone link](https://anvil.works/build#clone:X65WW2MQ5KCRTDJR=2PLIEVQGE4V6HQJG53PBVHDY).

## Validate inputs

Run the local tests without starting Anvil or loading TensorFlow:

```bash
python -m unittest discover -s tests -v
```

The tests exercise the bundled sample CSV and malformed, non-finite, and
oversized inputs.
