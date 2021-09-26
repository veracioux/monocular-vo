# Monocular visual odometry

## Setup

The project uses [this KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
for validation. First you will have to to place the downloaded set in the `data/`
directory, achieving the following directory structure:

```
data/
├── calib/
│   ├── 00
│   ├── 01
│   └── ...
├── poses/
│   ├── 00.txt
│   ├── 01.txt
│   └── ...
└── sequences/
    ├── 00
    ├── 01
    └── ...
```

The directories can also be symlinks (that is encouraged actually).

## Running

Once you have set everything up properly, you can run the validation by running
`./main.py`. To see which options you have available, run

```
./main.py -h
```

## Authors

- Anes Hadžić
- Haris Gušić
