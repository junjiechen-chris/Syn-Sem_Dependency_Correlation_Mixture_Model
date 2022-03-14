import json
import re
import sys
import nltk
import pandas as pd
# from nested_csv import generate_fieldnames
file_names_list = sys.argv[3:]
mode = sys.argv[2]

def extract_test_result_pair(fn, lines):
  test_result_list = []
  for idx in range(len(lines)):
    # print(lines[idx])
    if lines[idx].startswith('INFO:tensorflow:Evaluating on'):
      test_file = re.search(r"INFO:tensorflow:Evaluating .+\['.+/([a-z\-]+)\..+'\]", lines[idx]).group(1)
      eval_cont = json.loads(lines[idx+1][16:])
      eval_cont['alias'] = "{}_{}".format(fn, test_file)
      test_result_list.append(eval_cont)
  return test_result_list






if mode == "transformation_analysis":
  print("start analyzing")
  container = []
  for file_name in file_names_list:
    print(file_name)
    with open(file_name) as f:
      lines = f.readlines()
      lines = [line.strip() for line in lines]
    container.extend(extract_test_result_pair(file_name, lines))
    # print(extract_test_result_pair(lines))
    # print(container)
  # data = pd.DataFrame.from_dict(container)
  print(container)
  data = pd.json_normalize(container, max_level=1)
  print(data)
  header = ['alias', 'srl_f1.original', 'srl_f1.fix_labels', 'srl_f1.move_arg', 'srl_f1.merge_spans', 'srl_f1.split_spans', 'srl_f1.fix_boundary',
            'srl_f1.drop_overlapping_arg', 'srl_f1.drop_arg', 'srl_f1.add_arg']
  data.to_csv(sys.argv[1], header=header, columns=header)
elif mode == "dev_scores":
  print("start analyzing")
  container = []
  for file_name in file_names_list:
    print(file_name)
    with open(file_name) as f:
      line = f.readline()
      result = json.loads(line)
      print(result)
      eval_dict = {'alias': file_name, 'srl_f1': result[0]['score']}
      container.append(eval_dict)
  data = pd.json_normalize(container, max_level=1)
  header = ['alias', 'srl_f1']
  data.to_csv(sys.argv[1], header=header, columns=header)
else:
  raise NotImplementedError
#
# data = data.reindex(sorted(data.columns), axis=1)
# data.to_csv(sys.argv[1])


# with open(sys.argv[-1], "w") as f:

print(data)
