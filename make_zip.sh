#!/usr/bin/env bash


main()
{
    shopt -s globstar
    shopt -s nullglob
    files=("task.sh" "src"/**/*.py)

    if [[ ! -e "report/report.pdf" ]]; then
        (
            cd report
            pdflatex report.tex
        )
   fi

    if [[ ! -e "report/report.pdf" ]]; then
        exit 1
    fi

    files+=("report/report.pdf")
    zip "heng_julian_19473701.zip" "${files[@]}"
}

main
