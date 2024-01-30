#!/bin/bash
# generate tiddlers per image
help=$(
cat << EOF
usage: $0 <filename>
example: $0 ../png/sequential.png
EOF
)

if [ $# -eq '0' ]; then
   echo "$help";  
   exit
fi
## for file in `ls -1 ../png/*png`
## do
   fname=`basename $1|sed 's/\.png$//'`;
   cat << EOF
<div title="ti_$fname" creator="abhi" modifier="abhi" created="201511291511" modified="201511291536" tags="tikzimg figure" changecount="10">
<pre>&lt;&lt;tikzimg &quot;$fname.png&quot;&gt;&gt;</pre>
</div>
EOF
## done
