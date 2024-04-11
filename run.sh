#!/bin/bash

echo "Run handin template"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist, creating it!"
  mkdir plots
fi

# Download files from given links if they are not already present
echo "Downloading files..."
#wget -nc https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m11.txt
#wget -nc https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m12.txt
#wget -nc https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m13.txt
#wget -nc https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m14.txt
#wget -nc https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m15.txt

# Run satellite.py script
echo "Run the Satellite script ..."
python3 satellite.py

echo "Generating the PDF"

#pdflatex hand-in-3.tex
#bibtex hand-in-3.aux
#pdflatex hand-in-3.tex
#pdflatex hand-in-3.tex


echo "Script execution completed."
