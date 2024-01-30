#!/bin/bash
grep -o 'title="ti_[^"]*" ' ~/Dropbox/proj/tiddly/notes.html | sed -e 's/title="ti_//' -e 's/" /.png/' | sort > .to_be_deleted.tiddlyList;
ls -1 ../png/ | sort > .to_be_deleted.tikzList;
filesToAdd=`gcomm -23 .to_be_deleted.tikzList .to_be_deleted.tiddlyList`;

for f in $filesToAdd
do
   tiddly_tikz.sh ../png/$f
done
