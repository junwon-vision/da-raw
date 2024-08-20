import fiftyone as fo
import fiftyone.zoo as foz
import pickle
import os


def main():
    name = "my-dataset"
    
    ### foggy cityscape    
    # dataset_dir = "../datasets/test_data"
    # desp_list = ['pkl_sup.pkl',
    #              'pkl_0428_output_foggy_image_bs4.pkl',
    #              'pkl_0428_output_foggy_imginst_bs1.pkl',]
    
    ### bdd 100k
    # dataset_dir = "../datasets/test_data_bdd"
    # desp_list = [
    #     'pkl_0815_bdd_suponly_model_0011499.pkl',
    #     'pkl_0831_synthetic_augment_after7500_model_0011499.pkl',
    #     'pkl_0822_DAFRCNN_bddrain_all_model_0014999.pkl',
    #     'pkl_0904_bddrain_SWDA_model_0014999.pkl',
    #     'pkl_0814_3_bdd_pcp_image_pxllevel_CBAM_withsrc_model_0011499.pkl',
    # ]
    
    ### taodac rain
    # dataset_dir = "../datasets/test_data_taodac_rain"
    # desp_list = [
    #     'pkl_0815_bdd_suponly_model_0011499.pkl',
    #     'pkl_0831_synthetic_augment_after7500_model_0011499.pkl',
    #     'pkl_0822_DAFRCNN_taodacrain_all_model_0014999.pkl',
    #     'pkl_0904_taodacrain_SWDA_model_0014999.pkl',
    #     'pkl_0818_taodac_pcp_image_bs8_model_0011499.pkl',
    # ]
    
    ### bdd snow
    dataset_dir = "../datasets/test_data_bdd_snow"
    desp_list = [
        'pkl_0815_bdd_suponly_model_0011499.pkl',
        'pkl_0831_synthetic_augment_after7500_model_0011499.pkl',
        'pkl_0822_DAFRCNN_bddsnow_all_model_0014999.pkl',
        'pkl_0904_bddsnow_SWDA_model_0014999.pkl',
        'pkl_0821_bdd_snow_pcp_image_model_0011499.pkl',
    ]
    
    ### taodac snow
    # dataset_dir = "../datasets/test_data_taodac_snow"
    # desp_list = [
    #     'pkl_0815_bdd_suponly_model_0011499.pkl',
    #     'pkl_0831_synthetic_augment_after7500_model_0011499.pkl',
    #     'pkl_0822_DAFRCNN_taodacsnow_all_model_0014999.pkl',
    #     'pkl_0904_taodacsnow_SWDA_model_0014999.pkl',
    #     'pkl_0821_taodac_snow_pcp_image_model_0011499.pkl',
    # ]


    # Create the dataset
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.VOCDetectionDataset,
        name=name,
    )

    # View summary info about the dataset
    print(dataset)
    
    prediction_list = []
    for i in range(len(desp_list)):
        pkl_dir = os.path.join('./pkl_files',desp_list[i])
        with open (pkl_dir, 'rb') as fp:
            predictions = pickle.load(fp)
            prediction_list.append(predictions)


    for sample in dataset:
        filepath = sample.filepath

        # Convert predictions to FiftyOne format
        filepath = os.path.basename(filepath)[:-4]
        # 
        for index, predictions in enumerate(prediction_list):
            detections = []
            for obj in predictions[filepath]:
                label = obj["label"]
                confidence = obj["score"]
                # confidence = obj["prop_score"]
                # Bounding box coordinates should be relative values
                # in [0, 1] in the following format:
                # [top-left-x, top-left-y, width, height]
                # 
                bounding_box = obj['bbox']

                detections.append(
                    fo.Detection(
                        label=label,
                        bounding_box=bounding_box,
                        confidence=confidence,
                    )
                )
            # 

            # Store detections in a field name of your choice
            sample["predictions_{}".format(index)] = fo.Detections(detections=detections)

            # indent or not?
            sample.save()

    # Print the first few samples in the dataset
    session = fo.launch_app(dataset)
    session.wait()

if __name__ == '__main__':
    main()