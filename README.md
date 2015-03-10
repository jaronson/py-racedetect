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

Your colorferet (or whatever) images should be in a format as such:
```
data/
├── ground_truths/
│   └── name_value/
│       ├── 001/
│       │   ├── 001_fa_a.txt
│       │   └── 001_fb_a.txt
│       └── 002/
│           └── 002_fa_a.txt
└── images/
    ├── 001/
    │   ├── 001_fa_a.ppm.bzip2
    │   └── 001_fb_a.ppm.bzip2
    └── 002/
        └── 002_fa_a.ppm.bzip2
```

Use a crappy script like this or write a better one to format your data.
```
Dir.glob('dvd1/data/images/*').each do |dir|
  Dir.glob("#{dir}/*").each do |file|
    base = file.split('.').first
    ppm  = "#{base}.ppm"
    png  = "#{base}.png"

    if file =~ /bz2$/
      `bzip2 -d #{file}`
      `convert #{ppm} #{png}`
    elsif file =~ /ppm$/
      `convert #{ppm} #{png}`
      `rm #{ppm}`
    end
  end
end
```
