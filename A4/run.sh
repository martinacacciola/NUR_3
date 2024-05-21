#!/bin/bash

echo "Run handin template"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist, creating it!"
  mkdir plots
fi

# Run satellite.py script
echo "Run the Solar System script ..."
python3 solar_ss.py

echo "Generating the PDF"

pdflatex hand-in-4.tex
bibtex hand-in-4.aux
pdflatex hand-in-4.tex
pdflatex hand-in-4.tex

echo "Script execution completed."