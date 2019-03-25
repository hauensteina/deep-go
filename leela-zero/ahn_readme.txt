
readme.txt for leela-zero in deep-go
--------------------------------------

We use a local copy of leela-zero to generate training games.
The model used is build/elfv1 .
Get it from AWS S3:
$ cd build
$ aws s3 cp s3://ahn-uploads/elfv1 .

Generate training games with

$ validation/validation -g 1 -k sgf -n elfv1 -o '-g -p 1 --noponder -t 1 -d -r 0 -w'   -n elfv1 -o '-g -p 1 --noponder -t 1 -d -r 0 -w'


