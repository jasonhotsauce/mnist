import argparse
from os import path

import coremltools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help="The h5 file of the model from Keras")
    parser.add_argument('-o', '--output', help="The output file path")
    args = parser.parse_args()
    _, extension = path.splitext(args.model)
    if extension != '.h5':
        print("Only support h5 model file for now.")
        exit(1)
    author = input("Author: ")
    description = input("Description: ")
    license = input("license")
    input_name = input("Model input name: ")
    input_description = input("Model input description: ")
    output_name = input("Model output name: ")
    output_description = input("Model output description: ")
    class_labels_str = input("Class Labels (separate by ,): ")

    output_labels = class_labels_str.split(',')

    coreml_model = coremltools.converters.keras.convert(args.model,
                                                        input_names=input_name,
                                                        output_names=output_name,
                                                        image_input_names=input_name,
                                                        class_labels=output_labels,
                                                        image_scale=1/255.)
    coreml_model.author = author
    coreml_model.short_description = description
    coreml_model.license = license
    coreml_model.input_description[input_name] = input_description
    coreml_model.output_description[output_name] = output_description

    coreml_model.save(args.output)


if __name__ == '__main__':
    main()
