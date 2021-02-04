if __name__ == "__main__":
    # for y in modelTest(xTest):
    #     print(y.shape)
    
    # device = torch.device('cpu')
    # predTest = torch.rand((32,255,7,7))
    
    # # 1. Retrieve the outputs
    # outputs = split_output(predTest, device)
    
    # # 2. Get boxes from prediction
    # outputs[0] = prediction_to_boxes(outputs[0], 's_scale', (224,224))
    # # 3. Center boxes to corners
    # outputs[0] = boxes_center_to_corners(outputs[0])
    
    # # Filter the boxes
    # filtered_outputs = filter_boxes(*outputs)
    # print(filtered_outputs.shape)
    
    # # 4. Scale the boxes
    # # outputs[0] = scale_boxes(outputs[0], 224, 224)
    
    # # # Apply nms
    # # final_outputs = non_max_suppression(*outputs)