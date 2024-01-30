#!/bin/bash
## FOLDER=./.tikz
## rm -r $FOLDER;
## mkdir $FOLDER;
## cd $FOLDER;
## files=`find $HOME -iname "*.tex" | xargs -I {} grep -l "usetikzlibrary" {} | xargs -I {} grep -l "{standalone}" {}`;
## for f in $files
## do
##    ((x+=1));
##    dest=${x}.tex;
##    echo $f;
##    sed "/end{document}/i {$dest}" $f > $dest;
##    pdflatex $dest;
## done

FOLDER=~/Dropbox/proj/tikz/tex;
pushd ./
cd $FOLDER;
rm *pdf;
rm *_named.tex;
files=`ls -1 *tex`;
for f in $files
do
   echo "### $f";
   ### filenames
   png=`echo $f | sed -e "s/tex/png/"`;

   ## bypass pdf generation if png exists; remove all pngs to start afresh
   if [ -a "../png/$png" ]; then
      echo "$png exists, skipping pdf/png generation ...";
      continue;
   fi

   ### convert tex to pdf
   echo "converting to pdf ..."
   ## fname=`echo $f | sed -e 's/\(.*\)\.tex$/\1_named/'`;
   fname=`echo $f | sed -e 's/\(.*\)\.tex$/\1/'`;
   dest=${fname}.tex;
   ## flatex=`echo $f | sed -e 's/_/\\\\\\\_/g'`;
   ## sed -e "s/\(end{tikzpicture}\)/node[rectangle,fill=white,text=black,opacity=.75,draw=none] at (current bounding box.north west) {{\\\huge $flatex}};\n\\\\\1/" $f>$dest;
   pdflatex -interaction nonstopmode -halt-on-error -file-line-error $dest > /dev/null;
   fList="$fList ${fname}.pdf";
   echo "converting to png ..."
   convert -density 300 ${fname}.pdf ../png/$png
done

### wrapping up
pdftk $fList cat output ../figs.pdf;
mv *_named.pdf ../pdf/
rm *tmp;
rm *_named.tex;
clean_latex;
