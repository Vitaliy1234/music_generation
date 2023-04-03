import os

import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel

from miditok import REMI

from sample import midi_to_text


PPLM_BOW = 1
BOW_AFFECT = 4

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3

VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}

MODEL_FILE = os.path.join('..', 'models', 'el_yellow_ts.pt')
BAG_OF_BARS_FILE = os.path.join('linear_clf', 'bar_weights.xlsx')


def run_pplm_example(pretrained_model="gpt2-medium",
                     priming_sample="",
                     affect_weight=0.2,
                     knob=None,
                     uncond=False,
                     num_samples=1,
                     bag_of_words=None,
                     bag_of_words_affect=None,
                     discrim_weights=None,
                     discrim_meta=None,
                     class_label=-1,
                     length=100,
                     stepsize=0.02,
                     temperature=1.0,
                     top_k=10,
                     sample=True,
                     num_iterations=3,
                     grad_length=10000,
                     horizon_length=1,
                     window_length=0,
                     decay=False,
                     gamma=1.5,
                     gm_scale=0.9,
                     kl_scale=0.01,
                     seed=0,
                     no_cuda=False,
                     colorama=False,
                     verbosity='regular',
                     ):
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

    # load pretrained model and tokenizer
    model, tokenizer = load_model()
    model.to(device)
    model.eval()

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
        model=model,
        tokenizer=tokenizer,
        affect_weight=affect_weight,
        knob=knob,
        device=device,
        num_samples=num_samples,
        bag_of_words=bag_of_words,
        bag_of_words_affect=bag_of_words_affect,
        class_label=class_label,
        length=length,
        stepsize=stepsize,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        num_iterations=num_iterations,
        grad_length=grad_length,
        horizon_length=horizon_length,
        window_length=window_length,
        decay=decay,
        gamma=gamma,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        verbosity_level=verbosity_level
    )


def load_model():
    tokenizer = REMI()
    model = torch.jit.load(MODEL_FILE)

    return model, tokenizer


def generate_text_pplm(
        model,
        tokenizer,
        affect_weight=0.2,
        context=None,
        past=None,
        device="cuda",
        perturb=True,
        bow_indices=None,
        bow_indices_affect=None,
        affect_int=None,
        knob=None,
        classifier=None,
        class_label=None,
        loss_type=0,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR
):
    output_so_far = None

    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t

    # collect one hot vectors for bags of words
    one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices, tokenizer, device)
    affect_int_orig = affect_int
    one_hot_bows_affect, affect_int = build_bows_one_hot_vectors_aff(bow_indices_affect, affect_int, tokenizer, device)
#    print(torch.FloatTensor(one_hot_bows_affect).size())
    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []

    if verbosity_level >= VERBOSE:
        range_func = trange(length, ascii=True)
    else:
        range_func = range(length)
    count = 0
    int_score = 0
    for i in range_func:
        if count == 3:
          break
        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            if output_so_far.shape[1] > 1:
                _, past, _ = model(output_so_far[:, :-1])

        unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
        unpert_last_hidden = unpert_all_hidden[-1]

        # check if we are abowe grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past

        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past,
                    model,
                    last,
                    affect_weight = affect_weight,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    one_hot_bows_vectors=one_hot_bows_vectors,
                    one_hot_bows_affect=one_hot_bows_affect,
                    affect_int = affect_int,
                    knob = knob,
                    classifier=classifier,
                    class_label=class_label,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    device=device,
                    verbosity_level=verbosity_level
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past

        pert_logits, past, pert_all_hidden = model(last, past=pert_past)
        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
        pert_probs = F.softmax(pert_logits, dim=-1)

        if classifier is not None:
            ce_loss = torch.nn.CrossEntropyLoss()
            prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
            label = torch.tensor([class_label], device=device,
                                 dtype=torch.long)
            unpert_discrim_loss = ce_loss(prediction, label)
            if verbosity_level >= VERBOSE:
                print(
                    "unperturbed discrim loss",
                    unpert_discrim_loss.data.cpu().numpy()
                )
        else:
            unpert_discrim_loss = 0

        # Fuse the modified model and original model
        if perturb:

            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = ((pert_probs ** gm_scale) * (
                    unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST
            pert_probs = top_k_filter(pert_probs, k=top_k,
                                      probs=True)  # + SMALL_CONST

            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)
            # print('pert_prob, last ', pert_probs, last)

        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        output_so_far = (
            last if output_so_far is None
            else torch.cat((output_so_far, last), dim=1)
        )
        if verbosity_level >= REGULAR:
            print(tokenizer.decode(output_so_far.tolist()[0]))
        if tokenizer.decode(output_so_far.tolist()[0])[-1] == '.':
            count = count+1
        if bow_indices_affect is not None and [output_so_far.tolist()[0][-1]] in bow_indices_affect[0]:
            int_word = affect_int_orig[bow_indices_affect[0].index([output_so_far.tolist()[0][-1]])]
            print(tokenizer.decode(output_so_far.tolist()[0][-1]), int_word)
            int_score = int_score + int_word
    print("int_score: ", int_score)
    # print("int.. " , output_so_far.tolist()[0][-1])
    return output_so_far, unpert_discrim_loss, loss_in_time


def full_text_generation(
        model,
        tokenizer,
        affect_weight=0.2,
        knob=None,
        context=None,
        num_samples=1,
        device="cuda",
        bag_of_words=None,
        bag_of_words_affect=None,
        class_label=None,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR,
        **kwargs
):
    bow_indices = []
    bow_indices_affect = []

    if bag_of_words_affect:
        affect_words, affect_int = get_affect_words_and_int(bag_of_words_affect)
        bow_indices_affect.append([tokenizer.encode(word.strip(), add_prefix_space=True, add_special_tokens=False) for word in affect_words])

    loss_type = PPLM_BOW
    if bag_of_words_affect:
        loss_type = BOW_AFFECT

    unpert_gen_tok_text, _, _ = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        context=context,
        device=device,
        length=length,
        sample=sample,
        perturb=False,
        verbosity_level=verbosity_level
    )
    #
    # if device == 'cuda':
    #     torch.cuda.empty_cache()
    #
    # pert_gen_tok_texts = []
    # discrim_losses = []
    # losses_in_time = []
    # print("After Perturbation")
    # for i in range(num_samples):
    #     pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
    #         model=model,
    #         tokenizer=tokenizer,
    #         affect_weight=affect_weight,
    #         context=context,
    #         device=device,
    #         perturb=True,
    #         bow_indices=bow_indices,
    #         bow_indices_affect=bow_indices_affect,
    #         affect_int = affect_int,
    #         knob = knob,
    #         classifier=classifier,
    #         class_label=class_id,
    #         loss_type=loss_type,
    #         length=length,
    #         stepsize=stepsize,
    #         temperature=temperature,
    #         top_k=top_k,
    #         sample=sample,
    #         num_iterations=num_iterations,
    #         grad_length=grad_length,
    #         horizon_length=horizon_length,
    #         window_length=window_length,
    #         decay=decay,
    #         gamma=gamma,
    #         gm_scale=gm_scale,
    #         kl_scale=kl_scale,
    #         verbosity_level=verbosity_level
    #     )
    #     pert_gen_tok_texts.append(pert_gen_tok_text)
    #     if classifier is not None:
    #         discrim_losses.append(discrim_loss.data.cpu().numpy())
    #     losses_in_time.append(loss_in_time)
    #
    # if device == 'cuda':
    #     torch.cuda.empty_cache()
    #
    # return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
    return None, None, None, None


def build_bows_one_hot_vectors_aff(bow_indices, affect_int, tokenizer, device='cuda'):
    if bow_indices is None or affect_int is None:
        return None, None

    one_hot_bows_vectors = []
    # print(np.array(bow_indices).shape)
    for single_bow in bow_indices:
        zipped = [[single_bow[i], affect_int[i]] for i in range(len(single_bow))]
        single_bow_int = list(filter(lambda x: len(x[0]) <= 1, zipped))
        single_bow = [single_bow_int[i][0] for i in range(len(single_bow_int)) ]
        affect_ints = [single_bow_int[i][1] for i in range(len(single_bow_int)) ]
        # print(single_bow, affect_ints)
        # print(len(single_bow), len(affect_ints))
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        # print(num_words)
        one_hot_bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)
    return one_hot_bows_vectors, affect_ints


def get_affect_words_and_int(emotion):
    """
    Нужно получить такты, соответствующие эмоциям
    :param emotion: конкретная эмоция, такты для который ищем
    :return:
    """
    # filepath = cached_path(emotions)
    bag_of_bars = pd.read_excel(BAG_OF_BARS_FILE, index_col=0)
    bag_of_bars = bag_of_bars.sort_values(by=emotion)

    bars = bag_of_bars.tail(100)

    return list(bars.index), list(bars[emotion].values)


def generate(priming_sample):
    topics = [
        'legal']  # ,'military','monsters','politics','positive_words', 'religion', 'science','space','technology']
    emotions = ['cheerful']
    knob_vals = [0.8]  # ,0.5,0.7,1]

    for topic in topics:
        for emotion in emotions:
            for knob in knob_vals:
                print("topic:", topic, ", emotion:", emotion, ", knob is:", knob)
                run_pplm_example(
                    affect_weight=1,  # it is the convergence rate of emotion loss, don't change it :-p
                    knob=knob,  # 0-1, play with it as much as you want
                    priming_sample=priming_sample,
                    num_samples=1,
                    bag_of_words=topic,
                    bag_of_words_affect=emotion,
                    length=50,
                    stepsize=0.005,  # topic, emotion convergence rate
                    sample=True,
                    num_iterations=10,
                    window_length=5,
                    gamma=1.5,
                    gm_scale=0.95,
                    kl_scale=0.01,
                    verbosity='quiet'
                )


if __name__ == '__main__':
    # filename = os.path.join('..', 'data', 'jingle_bells.mid')
    # split_path = os.path.split(filename)
    # priming_txt = f"{split_path[1].split('.')[0]}.txt"
    # priming_txt = os.path.join('text_representations', priming_txt)
    # midi_to_text(filename, priming_txt)
    #
    # with open(priming_txt, 'r') as hfile:
    #     sample = hfile.read()
    sample = 'PIECE_START'
    generate(sample)
