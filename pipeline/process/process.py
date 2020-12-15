import apache_beam as beam
import argparse
import re
from collections import Counter
from apache_beam.options.pipeline_options import PipelineOptions

PROJECT = 'hamoye-296618'
BUCKET = 'used-cars'
DATASET = 'used_cars'

def extract_from_description(element):
    array = element.split(",")
    new_array = [*array[0:20],
                 str(" ".join(array[20:-3])).lower(), *array[-3:]]

    if len(new_array[15]) < 1 or new_array[15] == "":
        replacement = Counter(re.findall(
            r'4wd|fwd|rwd', new_array[20])).most_common(1)
        if len(replacement) > 0:
            new_array[15] = replacement[0][0]

    if len(new_array[17]) < 1 or new_array[17] == "":
        replacement = Counter(re.findall(
            r'suv|bus|convertible|coupe|hatchback|mini-van|offroad|pickup|sedan|truck|van|wagon', new_array[20])).most_common(1)
        if len(replacement) > 0:
            new_array[17] = replacement[0][0]

    if len(new_array[18]) < 1 or new_array[18] == "":
        replacement = Counter(re.findall(
            r'black|blue|brown|custom|green|grey|orange|purple|red|silver|white|yellow|gray', new_array[20])).most_common(1)
        if len(replacement) > 0:
            new_array[18] = replacement[0][0]

    if len(new_array[6]) < 1 or new_array[6] == "":
        replacement = Counter(re.findall(
            r'volvo|volkswagen|toyota|tesla|subaru|saturn|rover|ram|porsche|pontiac|nissan|morgan|mitsubishi|mini|mercury|mercedes-benz|mazda|lincoln|lexus|land rover|kia|jeep|jaguar|infiniti|hyundai|honda|hennessey|harley-davidson|gmc|ford|fiat|ferrari|dodge|datsun|chrysler|chevrolet|cadillac|buick|bmw|audi|aston-martin|alfa-romeo|acura', new_array[20])).most_common(1)
        if len(replacement) > 0:
            new_array[6] = replacement[0][0]
    


    return new_array

def filter_unwanted(element):

    try:
        return (
            isinstance(float(element[4]), float)
            and isinstance(float(element[11]), float)
            and float(element[4]) > 0.0
            or '' not in set(element)
            )
    except ValueError:
        return False


def process(element):
    return {
        'id': element[0],
        'region': element[2],
        'price': float(element[4]) if element[4].isdigit() else None,
        'year': element[5],
        'manufacturer': element[6] if bool(element[6]) else "unknown",
        'model': element[7] if bool(element[7]) else "unknown",
        'condition': element[8] if bool(element[8]) else "unknown",
        'cylinders': element[9] if bool(element[9]) else "unknown" ,
        'fuel': element[10] if bool(element[10]) else "unknown" ,
        'odometer': float(element[11]) if element[11].isdigit() else None,
        'title_status': element[12] if bool(element[12]) else "unknown" ,
        'transmission': element[13] if bool(element[13]) else "unknown",
        'drive': element[15] if bool(element[15]) else "unknown" ,
        'size': element[16] if bool(element[16]) else "unknown",
        'type': element[17] if bool(element[17]) else "unknown",
        'paint_color': element[18] if bool(element[18]) else "unknown",
        'state': element[-3] if bool(element[-3]) else "unknown",
    }





def process_two(element):
    return {
        'id': element[0],
        'region_url': element[3],
        'vin': element[14],
        'image_url': element[19],
        'description': element[20],
        'latitude': float(element[-2]) if element[-2] else None,
        'longitude': float(element[-1]) if element[-1] else None
    }


def run():
    argv = [
        '--project={0}'.format(PROJECT),
        '--job_name=examplejob1',
        '--save_main_session',
        '--staging_location=gs://{}/batch/staging'.format(BUCKET),
        '--temp_location=gs://{}/batch/temp'.format(BUCKET),
        '--region=us-central1',
        '--runner=DataflowRunner'
    ]

    with beam.Pipeline(argv=argv) as p:
        csv_lines = (
            p
            | "ReadDataFromGCS" >> beam.io.ReadFromText('gs://{}/data/vehicles.csv'.format(BUCKET), skip_header_lines=1)
            | "ExtractFromDescription" >> beam.Map(extract_from_description)
            | "Filter" >> beam.Filter(filter_unwanted)
        )

        schema_1 = 'id:STRING,region:STRING,price:FLOAT,year:STRING,manufacturer:STRING,model:STRING,condition:STRING,cylinders:STRING,fuel:STRING,odometer:FLOAT,title_status:STRING,transmission:STRING,drive:STRING,size:STRING,type:STRING,paint_color:STRING,state:STRING'

        schema_2 = 'id:STRING,region_url:STRING,vin:STRING,image_url:STRING,description:STRING,latitude:FLOAT,longitude:FLOAT'

        (
            csv_lines
            | "Separate StructuedDataSource" >> beam.Map(process)
            | "Write Structured to BigQuery" >> beam.io.WriteToBigQuery(
                f'{PROJECT}:{DATASET}.vehicles_min',
                schema=schema_1,
                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            )
        )

        (
            csv_lines
            | "Separate UntructuedDataSource" >> beam.Map(process_two)
            | "Write Unstructured to BigQuery" >> beam.io.WriteToBigQuery(
                f'{PROJECT}:{DATASET}.unstructured_sources_min',
                schema=schema_2,
                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            )
        )

        result = p.run()
        result.wait_until_finish()


if __name__ == '__main__':
    run()
