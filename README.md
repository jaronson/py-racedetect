# race-detect

The core of an untitled interactive art installation.

## Installation
```
brew install opencv
export PYTHONPATH=/usr/local/Cellar/opencv/2.4.9/include/python2.7:$PYTHONPATH
pip install -r requirements.txt
```

## Configuration
Copy `config/config.json.sample` to `config/config.json` and edit as appropriate.

## Data Setup

Your images should be in a format as such. If using FERET, use the provided `scripts/feret_mutator.rb` to convert.
```
mutated/
├── 00001
│  ├── 00001_930831_fa_a.png
│  ├── 00001_930831_fb_a.png
│  └── truths.json
├── 00002
│  ├── 00002_940128_fb.png
│  ├── 00002_940422_fa.png
│  ├── 00002_940422_fb.png
│  ├── 00002_940928_fa.png
│  ├── 00002_940928_fb.png
│  ├── 00002_940928_rb.png
│  ├── 00002_940928_rc.png
│  └── truths.json
```

## Testing
`python -m unittest discover`
