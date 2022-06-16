'''
Created on Jun 16, 2022

@author: yhu42
'''


import json
from neurotpr import geoparse

geoparse.load_model("C:/Users/yhu42/PretrainedModel/")


def run_neurotpr():
    inputTweet = "#HurricaneHarvey      #Harvey:    does anyone know about the flooding conditions around Cypress Ridge High School?! #HarveyFlood"
    raw_result = geoparse.topo_recog(inputTweet)
    print("Raw result is: "+raw_result)
    
    processed_result = correct_detected_location_desc_index(raw_result,inputTweet)
    print("Processed result is: "+str(processed_result))
   
    

# We found that the start and end indice of the detected location descriptions returned by the pretrained model were based on processed tweets not the original tweets
# The function below help get the original start and end indices of the detected location descriptions
def correct_detected_location_desc_index(raw_result_from_neurotpr, original_input_tweet):
    geoparse_result = json.loads(raw_result_from_neurotpr)
          
    processed_result = []
    unique_location_dict = {} # to get the correct start and end index
    for locDesc in geoparse_result:
        location_desc = locDesc["location_name"]
        if location_desc in unique_location_dict.keys():
            start_index = original_input_tweet.index(location_desc,unique_location_dict[location_desc])
        else:
            start_index = original_input_tweet.index(location_desc)
        
        end_index = start_index + len(location_desc)
        unique_location_dict[location_desc] = end_index
        
        processed_loc_desc = {"location_name":location_desc,"start_idx": start_index, "end_idx": end_index}
        processed_result.append(processed_loc_desc)
        
    return processed_result
        
           

    
    
if __name__ == '__main__':
    run_neurotpr()