#!/usr/bin/env bash


main()
{
    shopt -s globstar
    shopt -s nullglob
    files=("task.sh" "src"/**/*.py)

    report_fname="heng_julian_19473701_mp_assignment_2_report.pdf"
    report="report/${report_fname}"
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

    cp "${report}" "${report_fname}"

    files+=("${report_fname}")
    zip "heng_julian_19473701.zip" "${files[@]}"
}

main
