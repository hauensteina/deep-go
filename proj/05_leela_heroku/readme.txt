
A back end API to ask leela for moves and run a keras scoring network
========================================================================
AHN, April 2019

Used by the heroku app leela-one-playout, which lives in a separate repo
on heroku, not github.
 
To start the back end leela( *this* project, on github), use

gunicorn leela_server:app --bind 0.0.0.0:2718 -w 3

inside a tmux session, to keep it running.

For the heroku app,

cd leela-one-playout // wherever you cloned it to from heroku
git push heroku master.

=== The End ===

