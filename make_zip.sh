#!/usr/bin/env bash


main()
{
    shopt -s globstar
    shopt -s nullglob

    files=("task.sh" "src"/**/*.py)

    declaration="DeclarationOfOriginality_v1.1.pdf"
    report_fname="heng_julian_19473701_mp_assignment_2_report.pdf"
    report="report/report.pdf"
    if [[ ! -e "${report}" ]]; then
        (
            cd report
            bash ./gen_figures.sh
            bash ./make_report.sh
        )
    fi

    if [[ ! -e "${report}" ]]; then
        exit 1
    fi

    gs -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -sOUTPUTFILE="${report_fname}" "${declaration}" "${report}"

    files+=("${declaration}" "${report_fname}")
    zip "heng_julian_19473701.zip" "${files[@]}"
}

main
