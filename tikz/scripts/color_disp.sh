#!/bin/bash
#Takes as input a color list of the format "color_name html_code" and
#outputs a pdf file with colors.
#NOTE: html_code should not have a # at the beginning.
infile=$1;
outfileName=`basename $infile`;

#tikz preamble
cat << EOF > $outfileName.tex
\documentclass[tikz,border=2]{standalone}
\usepackage{lmodern} % enhanced version of computer modern
\usepackage[T1]{fontenc} % for hyphenated characters
EOF

#define colors
awk '/^#/{next}{print "\\definecolor{" $1 "}{HTML}{" $2 "}"}' $infile >> $outfileName.tex

#start tikz picture
cat << EOF >> $outfileName.tex
\begin{document}
\begin{tikzpicture}[
every node/.style={rectangle, minimum width=4cm, minimum height=.5cm, draw=none}]
EOF

#colors to nodes
awk 'BEGIN{y=0}/^#/{next}{print "\\node [label=left:" $1 ",fill=" $1 "] at (0," y "cm) {};"; y+=.75}' $infile >> $outfileName.tex

## postamble
cat << EOF >> $outfileName.tex
\end{tikzpicture}
\end{document}
EOF
#cat $outfileName.tex
pdflatex $outfileName.tex
