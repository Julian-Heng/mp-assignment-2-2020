#!/usr/bin/env bash


main()
{
    set -x
    base_dir="/home/student"
    test_dir="${base_dir}/test"
    train_dir="${base_dir}/train"
    val_dir="${base_dir}/val"

    extract_dir="${base_dir}/heng_julian_19473701"
    out_dir="${extract_dir}/output"

    module="${extract_dir}/src"
    classifier_output="${out_dir}/train.npz"

    shopt -s globstar
    shopt -s nullglob
    training_files=("${train_dir}"/[0-9]/**/*.{jpg,png})
    testing_files=("${test_dir}"/**/*.{jpg,png})
    val_files=("${val_dir}"/**/*.{jpg,png})
    shopt -u globstar
    shopt -u nullglob

    (
        cd "${module}"
        python3 -m mp_ocr --debug-log --train --train-output "${classifier_output}" "${training_files[@]}"
        #python3 -m mp_ocr --debug-log --classifier "${classifier_output}" --output "${out_dir}" "${val_files[@]}"
        python3 -m mp_ocr --debug-log --classifier "${classifier_output}" --output "${out_dir}" "${testing_files[@]}"
    )
    set +x
}

main
