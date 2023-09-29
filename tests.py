from Spam_Detector_Machine_Learning import posterior, learn_prior, learn_likelihood, nb_classify


def test_2():
    prior = 0.05
    likelihood = ((0.001, 0.3),(0.05,0.9),(0.7,0.99))

    observation = (True, True, True)

    class_posterior_true = posterior(prior, likelihood, observation)
    print("P(C=False|observation) is approximately {:.5f}"
        .format(1 - class_posterior_true))
    print("P(C=True |observation) is approximately {:.5f}"
        .format(class_posterior_true))
    
    print("")

    prior = 0.05
    likelihood = ((0.001, 0.3),(0.05,0.9),(0.7,0.99))

    observation = (True, False, True)

    class_posterior_true = posterior(prior, likelihood, observation)
    print("P(C=False|observation) is approximately {:.5f}"
        .format(1 - class_posterior_true))
    print("P(C=True |observation) is approximately {:.5f}"
        .format(class_posterior_true))
    
    print("")

    prior = 0.05
    likelihood = ((0.001, 0.3),(0.05,0.9),(0.7,0.99))

    observation = (False, False, True)

    class_posterior_true = posterior(prior, likelihood, observation)
    print("P(C=False|observation) is approximately {:.5f}"
        .format(1 - class_posterior_true))
    print("P(C=True |observation) is approximately {:.5f}"
        .format(class_posterior_true))  


    print("")

    prior = 0.05
    likelihood = ((0.001, 0.3),(0.05,0.9),(0.7,0.99))

    observation = (False, False, False)

    class_posterior_true = posterior(prior, likelihood, observation)
    print("P(C=False|observation) is approximately {:.5f}"
        .format(1 - class_posterior_true))
    print("P(C=True |observation) is approximately {:.5f}"
        .format(class_posterior_true))  
    

def test_3():
    prior = learn_prior("spam-labelled.csv")
    print("Prior probability of spam is {:.5f}.".format(prior))
    print("")
    prior = learn_prior("spam-labelled.csv")
    print("Prior probability of not spam is {:.5f}.".format(1 - prior))
    print("")
    prior = learn_prior("spam-labelled.csv", pseudo_count = 1)
    print(format(prior, ".5f"))
    print("")
    prior = learn_prior("spam-labelled.csv", pseudo_count = 2)
    print(format(prior, ".5f"))
    print("")
    prior = learn_prior("spam-labelled.csv", pseudo_count = 10)
    print(format(prior, ".5f"))
    print("")
    prior = learn_prior("spam-labelled.csv", pseudo_count = 100)
    print(format(prior, ".5f"))
    print("")
    prior = learn_prior("spam-labelled.csv", pseudo_count = 1000)
    print(format(prior, ".5f"))

def test_4():
    likelihood = learn_likelihood("spam-labelled.csv")
    print(len(likelihood))
    print([len(item) for item in likelihood])
    
    print("")

    likelihood = learn_likelihood("spam-labelled.csv")

    print("P(X1=True | Spam=False) = {:.5f}".format(likelihood[0][False]))
    print("P(X1=False| Spam=False) = {:.5f}".format(1 - likelihood[0][False]))
    print("P(X1=True | Spam=True ) = {:.5f}".format(likelihood[0][True]))
    print("P(X1=False| Spam=True ) = {:.5f}".format(1 - likelihood[0][True]))

    print("")

    likelihood = learn_likelihood("spam-labelled.csv", pseudo_count=1)

    print("With Laplacian smoothing:")
    print("P(X1=True | Spam=False) = {:.5f}".format(likelihood[0][False]))
    print("P(X1=False| Spam=False) = {:.5f}".format(1 - likelihood[0][False]))
    print("P(X1=True | Spam=True ) = {:.5f}".format(likelihood[0][True]))
    print("P(X1=False| Spam=True ) = {:.5f}".format(1 - likelihood[0][True]))

def test_5():
    prior = learn_prior("spam-labelled.csv")
    likelihood = learn_likelihood("spam-labelled.csv")

    input_vectors = [
        (1,1,0,0,1,1,0,0,0,0,0,0),
        (0,0,1,1,0,0,1,1,1,0,0,1),
        (1,1,1,1,1,0,1,0,0,0,1,1),
        (1,1,1,1,1,0,1,0,0,1,0,1),
        (0,1,0,0,0,0,1,0,1,0,0,0),
        ]

    predictions = [nb_classify(prior, likelihood, vector) 
                for vector in input_vectors]

    for label, certainty in predictions:
        print("Prediction: {}, Certainty: {:.5f}"
            .format(label, certainty))

    print("")

    prior = learn_prior("spam-labelled.csv", pseudo_count=1)
    likelihood = learn_likelihood("spam-labelled.csv", pseudo_count=1)

    input_vectors = [
        (1,1,0,0,1,1,0,0,0,0,0,0),
        (0,0,1,1,0,0,1,1,1,0,0,1),
        (1,1,1,1,1,0,1,0,0,0,1,1),
        (1,1,1,1,1,0,1,0,0,1,0,1),
        (0,1,0,0,0,0,1,0,1,0,0,0),
        ]

    predictions = [nb_classify(prior, likelihood, vector) 
                for vector in input_vectors]

    for label, certainty in predictions:
        print("Prediction: {}, Certainty: {:.5f}"
            .format(label, certainty))
        


def main():

    print("\nTest Set 2: ")
    test_2()
    print("\nTest Set 3: ")
    test_3()
    print("\nTest Set 4: ")
    test_4()
    print("\nTest Set 5: ")
    test_5()

main()