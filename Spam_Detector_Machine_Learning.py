import csv


def posterior(prior, likelihood, observation):
    # For Value is True
    pt = 1 * prior
    for i in range(len(observation)):
        if observation[i] == True:
            pt = pt * likelihood[i][1]

        else:
            pt = pt * (1 - likelihood[i][1])


    pf = 1 * (1 - prior)
    for i in range(len(observation)):
        if observation[i] == True:
            pf = pf * likelihood[i][0]
        else:
            pf = pf * (1 - likelihood[i][0])

    return pt/(pt+pf)

def learn_prior(file_name, pseudo_count=0):
    with open(file_name) as in_file:
        training_examples = [tuple(row) for row in csv.reader(in_file)] 


    t_count = 0
    f_count = 0
    for i in range(1, len(training_examples)):
        # print(training_examples[i][-1])
        if training_examples[i][-1] == "1":
            t_count += 1
        else:
            f_count += 1

    return (t_count + pseudo_count) / ((t_count + pseudo_count) + (f_count + pseudo_count))

def learn_likelihood(file_name, pseudo_count=0):
    

    with open(file_name) as in_file:
        training_examples = [tuple(row) for row in csv.reader(in_file)] 
    
    likelihood = []

    for i in range(12):
        true_count = 0
        false_count = 0
        true_total = 0
        false_total = 0

        for k in range(1, len(training_examples)):
            if training_examples[k][-1] == "1":
                true_total += 1
                if training_examples[k][i] == "1":
                    true_count += 1
            else:
                false_total += 1
                if training_examples[k][i] == "1":
                    false_count += 1
        

        likelihood.append(((false_count + pseudo_count)/(false_total + (2 * pseudo_count)), (true_count + pseudo_count)/(true_total + (2 * pseudo_count))))

    

    return likelihood

def nb_classify(prior, likelihood, input_vector):

    post = posterior(prior, likelihood, input_vector)

    if post <= 0.5:
        return ("Not Spam", 1-post)
    
    else:
        return ("Spam", post)
