.DEFAULT_GOAL:=main.pdf

.PHONY: clean
clean:
	git clean -xf *converted-to.pdf *.blg *.log *.aux mainNotes.bib

.PHONY: compress
compress: clean
	zip -r manuscript-$(shell date +%d%b).zip . -x "Makefile" -x "*.zip" -x ".*"

%.aux: %.tex
	pdflatex $<

%.bbl: %.aux *.bib
	bibtex $(basename $@)

%.pdf: %.tex %.bbl
	pdflatex $<
	pdflatex $<
