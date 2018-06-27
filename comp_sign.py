Thresholds

main_thr_1 = 950.0 # Normal
main_thr_2 = 800.0 # High
main_thr_3 = 700.0 # Very High

same_upper  = 1050.0
same_middle = 800.0
same_lower  = 600.0

forg_upper  = 1350.0
forg_middle = 1120.0
forg_lower  = 810.0

diff_upper  = 1800.0
diff_middle = 1345.0
diff_lower  = 1150.0

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
    decision = -1

    if level is 0:
        main_thr = main_thr_1
    elif level is 1:
        main_thr = main_thr_2
    elif level is 2:
        main_thr = main_thr_3

    if(dist<main_thr):
        decision = 1
    else:
        decision = 0

    same_per = 0.0
    forg_per = 0.0
    diff_per = 0.0

    # Calculating same_per
    if(dist<same_lower):
        same_per = 100 - ((dist-0)/(same_lower-0))*5.0
    elif(dist<same_middle):
        same_per = 95 - ((dist-same_lower)/(same_middle-same_lower))*45
    elif(dist<same_upper):
        same_per = 50 - ((dist-same_middle)/(same_upper-same_middle))*45
    elif(dist>1350):
        same_per = 0
    elif(dist>same_upper):
        same_per = 5 - ((dist-same_upper)/(1350-same_upper))*5

    # Calculating forg_per
    if((dist<forg_lower)&(dist>=700)):
        forg_per = ((dist-700)/(forg_lower-700))*15
    elif(dist<700):
        forg_per = 0.0
    elif(dist<forg_middle):
        forg_per = 15 + ((dist-forg_lower)/(forg_middle-forg_lower))*60
    elif(dist<forg_upper):
        forg_per = 15 + ((dist-forg_middle)/(forg_upper-forg_middle))*60
    elif(dist>=2000):
        forg_per = 0.0
    elif(dist>forg_upper):
        forg_per = ((dist-forg_upper)/(2000-forg_upper))*15

    # Calculating diff_per
    if(dist<=1000):
        diff_per = 0.0
    elif(dist<diff_lower):
        diff_per = ((dist-1000)/(diff_lower-1000))*5.0
    elif(dist<diff_middle):
        diff_per = 5 + ((dist-diff_lower)/(diff_middle-diff_lower))*45
    elif(dist<diff_upper):
        diff_per = 50 + ((dist-diff_middle)/(diff_upper-diff_middle))*45
    elif(dist>diff_upper):
        diff_per = 95 + ((dist-same_upper)/(3000-same_upper))*5

    if(dist>=3000):
        same_per = 0.0
        forg_per = 0.0
        diff_per = 100.0


    return dist,decision,same_per,forg_per,diff_per