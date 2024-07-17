import pandas as pd
import re
from uie_predictor import UIEPredictor


schema =  ['日期', '始发地', '目的地', '车次', '甩挂车号', '换挂车号', '装载物体', '物体重量']
ie = UIEPredictor(model='uie-base', schema=schema)


def extract_date(text):
    match = re.search(r'(\d+月\d+日)', text)
    return match.group(1) if match else None


def extract_train_number(text):
    match = re.search(r'[ZK]\d+(?:/\d+)?次', text)
    return match.group() if match else None


def extract_weight(text):
    match = re.search(r'(\d+(?:\.\d+)?)kg', text)
    return match.group(1) if match else None

def extract_cargo(text):
    match = re.search(r'装超重([\u4e00-\u9fa5]+)', text)
    return match.group(1) if match else None

def extract_destination(text):
    match = re.search(r'到(\w+(?:南|西|东|北)?站)', text)
    return match.group(1) if match else None



def extract_car_numbers(text):
    detached_car = re.search(r'甩(\w+\s*\d+)\s*\((\d+)(?:,[\w\s]+)?\)', text)
    attached_car = re.search(r'换挂(\w+\s*\d+)\s*\((\d+)(?:,[\w\s]+)?\)', text)

    detached_car = f"{detached_car.group(1)}({detached_car.group(2)})" if detached_car else ""
    attached_car = f"{attached_car.group(1)}({attached_car.group(2)})" if attached_car else ""

    return detached_car, attached_car


def convert_parsed_result(parsed_result, original_text):
    simplified_result = {}
    for key, value in parsed_result[0].items():
        simplified_result[key] = value[0]['text'] if value else None


    if '日期' not in simplified_result or not simplified_result['日期']:
        simplified_result['日期'] = extract_date(original_text)

    if '车次' not in simplified_result or not simplified_result['车次']:
        simplified_result['车次'] = extract_train_number(original_text)

    if '装载物体' not in simplified_result or not simplified_result['装载物体']:
        simplified_result['装载物体'] = extract_cargo(original_text)

    if '物体重量' not in simplified_result or not simplified_result['物体重量']:
        simplified_result['物体重量'] = extract_weight(original_text)

    if '目的地' not in simplified_result or not simplified_result['目的地']:
        simplified_result['目的地'] = extract_destination(original_text)

    simplified_result['始发地'] = simplified_result.get('始发地', '昆明')
    detached_car, attached_car = extract_car_numbers(original_text)
    simplified_result['甩挂车号'] = detached_car
    simplified_result['换挂车号'] = attached_car
    return simplified_result


def parse_excel(file_path):
    df = pd.read_excel(file_path)
    parsed_data = []

    for index, row in df.iterrows():
        command = row['CMD_COMBINE']
        parsed_command = ie(command)
        parsed_data.append(convert_parsed_result(parsed_command, command))

    parsed_df = pd.DataFrame(parsed_data)
    return parsed_df



input_file_path = 'output.xlsx'
parsed_df = parse_excel(input_file_path)


output_file_path = 'structuredData.xlsx'
parsed_df.to_excel(output_file_path, index=False)

print(f"解析后的数据已保存至 {output_file_path}")