
#Thresholds

main_thr_1 = 0.0 # Normal
main_thr_2 = 0.0 # High
main_thr_3 = 0.0 # Very High

same_upper  = 0.0
same_middle = 0.0
same_lower  = 0.0

forg_upper  = 0.0
forg_middle = 0.0
forg_lower  = 0.0

diff_upper  = 0.0
diff_middle = 0.0
diff_lower  = 0.0

# level 0 = Normal, 1 = High, 2 = Very High
def compare_signatures(path1,path2,level):

    canvas_size = (952, 1360)
    max1 = 0
    max2 = 0

    # Load the model
    model_weight_path = 'models/signet.pkl'
    #model = TF_CNNModel(signet, model_weight_path)
    model = TF_CNNModel(tf_signet, model_weight_path)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    original1 = imread(path1, flatten=1)
    processed1 = preprocess_signature(original1, canvas_size)

    original2 = imread(path2, flatten=1)
    processed2 = preprocess_signature(original2, canvas_size)

    feature_vector1 = model.get_feature_vector(sess,processed1)
    feature_vector2 = model.get_feature_vector(sess,processed2) 
    feature_vector1 = feature_vector1.T
    feature_vector2 = feature_vector2.T 

    dist = (abs(feature_vector1**2 - feature_vector2**2))**(0.5)
    #print(dist)

    for idx, val in enumerate(dist):
        if np.isnan(val):
            dist[idx] = 0

    dist = np.sum(dist)

    main_thr = 0.0
    if level is 0:
        main_thr = main_thr_1
    elif level is 1:
        main_thr = main_thr_2
    elif level is 3:
        main_thr = main_thr_3

    decision = -1

    if(dist<main_thr):
        decision = 1
    else:
        decision = 0

    same_per = 0.0
    forg_per = 0.0
    diff_per = 0.0




    

    return dist,decision,same_per,forg_per,diff_per