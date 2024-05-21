#!/bin/bash

echo "Run handin template"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist, creating it!"
  mkdir plots
fi

# Download files from given links if they are not already present
echo "Downloading files..."
wget -nc https://home.strw.leidenuniv.nl/~daalen/Handin_files/galaxy_data.txt

# Run script for first exercise
echo "Run the Solar System script ..."
python3 solar_ss.py

# Run script for second exercise
echo "Run the FFT script ..."
python3 fft.py

# Run script for third exercise
echo "Run the Learning script ..."
python3 learning.py

echo "Generating the PDF"

pdflatex hand-in-4.tex
bibtex hand-in-4.aux
pdflatex hand-in-4.tex
pdflatex hand-in-4.tex

echo "Script execution completed."