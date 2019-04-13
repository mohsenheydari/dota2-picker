# Dota 2 Picker

[![GitHub Stars](https://img.shields.io/github/stars/MohsenHeydari/dota2-picker.svg)](https://github.com/MohsenHeydari/dota2-picker/stargazers) [![GitHub Issues](https://img.shields.io/github/issues/MohsenHeydari/dota2-picker.svg)](https://github.com/MohsenHeydari/dota2-picker/issues) [![Current Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/MohsenHeydari/dota2-picker)

Dota 2 Picker is a tool that uses Machine Learning to find out which hero is picked in game and use that information to provide pick suggestions using "dotapicker.com/herocounter"

---

## Table of contents

- [Requirements](#requirements)
- [Usage](#usage)
- [Features](#features)
- [Limitations](#limitations)
- [License](#license)
- [Contact](#contact)

## Requirements

The project requires a few Python 3.7 packages. Install them using:

```bash
pip install -r requirements.txt
```

Currently it is only tested on Windows 10 OS and works on 16:9 aspect ratio

## Usage

After you clone this repo to your desktop, go to the root directory of the project and run using:

```bash
python app.py
```

## Features

- Uses convolutional neural networks to categorize heroes
- Displays a browser (using QT) in game to show pick suggesstions
- Does not modify any game data and its safe to use in game

## Limitations

- Currently not 100% accurate
- Needs more optimization

## License

> You can check out the full license [here](https://github.com/MohsenHeydari/dota2-picker/blob/master/LICENSE)

This project is licensed under the terms of the **GPL-3.0** license.

## Contact

Created by [Mohsen Heydari](http://venolabs.com/) - feel free to contact me!
