import apache_beam as beam
import argparse
import re
from apache_beam.options.pipeline_options import PipelineOptions 

PROJECT = 'hamoye-296618'
BUCKET = 'used-cars'
DATASET = 'used_cars_dataset'


def filter_unwanted(element):
    try:
        return (
            True if isinstance(float(element[4]), float)
            and isinstance(float(element[11]), float)
            and float(element[4]) > 0.0
            and len(element[9].split()) > 0
            else False)
    except ValueError:
        return False
    

def process(element):
    
    
    return {
        'id': element[0],
        'region': element[2],
        'price': float(element[4]) if element[4] else None,
        'year': element[5],
        'manufacturer': element[6],
        'model': element[7],
        'condition': element[8],
        'cylinders': int(re.findall(r'\d', element[9])[0]),
        'fuel': element[10],
        'odometer': float(element[11]) if element[11] else None,
        'title_status': element[12],
        'transmission': element[13],
        'drive': element[15],
        'size': element[16],
        'car_type': element[17],
        'paint_color': element[18],
        'state': element[-3],
        'latitude': float(element[-2]) if element[-2] else None,
        'longitude': float(element[-1]) if element[-1] else None
    }
    
def process_two(element):
    
    return {
        'id': element[0],
        'region_url': element[3],
        'vin': element[14],
        'image_url': element[19],
        'description': " ".join(element[20:-3])
    }
    
    
def run():
    argv = [
        '--project={0}'.format(PROJECT),
        '--job_name=examplejob',
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
            | "Split" >> beam.Map(lambda x: x.split(','))
            | "Filter" >> beam.Filter(filter_unwanted)
        )
        
        schema_1 = 'id:STRING,region:STRING,price:FLOAT,year:STRING,manufacturer:STRING,model:STRING,condition:STRING,cylinders:INTEGER,fuel:STRING,odometer:FLOAT,title_status:STRING,transmission:STRING,drive:STRING,size:STRING,car_type:STRING,paint_color:STRING,state:STRING,latitude:FLOAT,longitude:FLOAT'
                    
        schema_2 = 'id:STRING,region_url:STRING,vin:STRING,image_url:STRING,description:STRING'

        (
            csv_lines
            | "Separate StructuedDataSource" >> beam.Map(process)
            | "Write Structured to BigQuery" >> beam.io.WriteToBigQuery(
                f'{PROJECT}:{DATASET}.vehicles',
                schema=schema_1,
                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            )
        )
        
        (
            csv_lines
            | "Separate UntructuedDataSource" >> beam.Map(process_two)
            | "Write Unstructured to BigQuery" >> beam.io.WriteToBigQuery(
                f'{PROJECT}:{DATASET}.unstructured_sources',
                schema=schema_2,
                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            )
        )
        
        result = p.run()
        result.wait_until_finish()
        
if __name__ == '__main__':
    run()
