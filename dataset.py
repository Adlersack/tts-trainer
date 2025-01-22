import os, json
import shutil

def extractEnglishData(input, name):
    result = {}
    
    for hash in input:
        if hash[1]['fileName'][0] not in ['K', 'C', 'J']:
            if "npcName" in hash[1] and hash[1]["npcName"] == name:
                result[hash[0]] = hash[1]
    
    return result

# JSON-Structure:
# json[0] == hash + data
# json[0][0] == hash
# json[0][1] == data
# json[0][1][x] == data where x is 'fileName', 'language', 'npcName', 'text' or 'type'
# result_json = json.dumps(result, indent=4)
# hash_result = json.dumps(hash_result, indent=4)

if __name__ == "__main__":
    input = open("result_en.json", "r", encoding="utf-8").read()
    input = json.loads(input)

    name = 'Ganyu'
    name_folder_path = './characters/' + name
    name_audio_path = name_folder_path + '/audio'
    audio_files_path = './audio_files/'
    
    hash_result = extractEnglishData(input, name)
    hash_result = sorted(hash_result.items())
    
    if len(hash_result) > 0:
        os.makedirs(name_audio_path, exist_ok=True)
    
    print(f"Found {len(hash_result)} result(s) for '{name}'.")
    
    if os.path.exists(name_folder_path + '/metadata.txt') and os.path.getsize(name_folder_path + '/metadata.txt') != 0:
        print(f"Metadata.txt is not empty.")
    else:
        with open(name_folder_path + '/metadata.txt', 'w', encoding='utf-8') as metadata:
            for key in hash_result:
                audio = key[0] + ".wav"
                audio_path = f"{audio_files_path}/{audio}"
                
                if 'text' in key[1] and os.path.exists(audio_path):
                        meta_string = f"{key[0]}|{key[1]['text']}"
                        metadata.write(meta_string + "\n")
                        
                        shutil.move(audio_path, f"{name_audio_path}/{audio}")
                        print(f"Audio file '{key[0]}.wav' has been found and got moved to '{name_audio_path}/{audio}.'")
                        
    hash_result = json.dumps(hash_result, indent=4)
    open(os.path.join(name_folder_path, name + ".json"), "w", encoding="utf-8").write(hash_result)