import numpy as np

import copy
import os 
import gc 

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.data import *
from src.models import *
from src.fedavg import *
from src.client import * 
from src.clustering import *
from src.utils import * 

args = args_parser()

args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

torch.cuda.set_device(args.gpu) ## Setting cuda on GPU 

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass
    
path = args.savedir + args.alg + '/' + args.partition + '/' + args.dataset + '/'
mkdirs(path)

##################################### Data partitioning section 


args.local_view = True
X_train, y_train, X_test, y_test, net_dataidx_map, net_dataidx_map_test, \
traindata_cls_counts, testdata_cls_counts = partition_data(args.dataset, 
args.datadir, args.logdir, args.partition, args.num_users, beta=args.beta, local_view=args.local_view)

train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                   args.datadir,
                                                                                   args.batch_size,
                                                                                   32)

print("len train_ds_global:", len(train_ds_global))
print("len test_ds_global:", len(test_ds_global))

################################### Shared Data 
idxs_test = np.arange(len(test_ds_global))
labels_test = np.array(test_ds_global.target)
# Sort Labels Train 
idxs_labels_test = np.vstack((idxs_test, labels_test))
idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
idxs_test = idxs_labels_test[0, :]
labels_test = idxs_labels_test[1, :]

idxs_test_shared = []
N = args.nsamples_shared//args.nclasses
ind = 0
for k in range(args.nclasses): 
    ind = max(np.where(labels_test==k)[0])
    idxs_test_shared.extend(idxs_test[(ind - N):(ind)])

test_targets = np.array(test_ds_global.target)
for i in range(args.nclasses):
    print(f'Shared data has label: {i}, {len(np.where(test_targets[idxs_test_shared[i*N:(i+1)*N]]==i)[0])} samples')

shared_data_loader = DataLoader(DatasetSplit(test_ds_global, idxs_test_shared), batch_size=N, shuffle=False)

for x,y in shared_data_loader:
    print(x.shape)
    
################################### build model
def init_nets(args, dropout_p=0.5):

    users_model = []

    for net_i in range(-1, args.num_users):
        if args.dataset == "generated":
            net = PerceptronModel().to(args.device)
        elif args.model == "mlp":
            if args.dataset == 'covtype':
                input_size = 54
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'a9a':
                input_size = 123
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'rcv1':
                input_size = 47236
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'SUSY':
                input_size = 18
                output_size = 2
                hidden_sizes = [16,8]
            net = FcNet(input_size, hidden_sizes, output_size, dropout_p).to(args.device)
        elif args.model == "vgg":
            net = vgg11().to(args.device)
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(args.device)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).to(args.device)
            elif args.dataset == 'celeba':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2).to(args.device)
        elif args.model =="simple-cnn-3":
            if args.dataset == 'cifar100': 
                net = SimpleCNN_3(input_dim=(16 * 3 * 5 * 5), hidden_dims=[120*3, 84*3], output_dim=100).to(args.device)
            if args.dataset == 'tinyimagenet':
                net = SimpleCNNTinyImagenet_3(input_dim=(16 * 3 * 13 * 13), hidden_dims=[120*3, 84*3], 
                                              output_dim=200).to(args.device)
        elif args.model == "vgg-9":
            if args.dataset in ("mnist", 'femnist'):
                net = ModerateCNNMNIST().to(args.device)
            elif args.dataset in ("cifar10", "cinic10", "svhn"):
                # print("in moderate cnn")
                net = ModerateCNN().to(args.device)
            elif args.dataset == 'celeba':
                net = ModerateCNN(output_dim=2).to(args.device)
        elif args.model == 'resnet9': 
            if args.dataset == 'cifar100': 
                net = ResNet9(in_channels=3, num_classes=100)
        elif args.model == "resnet":
            net = ResNet50_cifar10().to(args.device)
        elif args.model == "vgg16":
            net = vgg16().to(args.device)
        else:
            print("not supported yet")
            exit(1)
        if net_i == -1: 
            net_glob = copy.deepcopy(net)
            initial_state_dict = copy.deepcopy(net_glob.state_dict())
            server_state_dict = copy.deepcopy(net_glob.state_dict())
            if args.load_initial:
                initial_state_dict = torch.load(args.load_initial)
                server_state_dict = torch.load(args.load_initial)
                net_glob.load_state_dict(initial_state_dict)
        else:
            users_model.append(copy.deepcopy(net))
            users_model[net_i].load_state_dict(initial_state_dict)

#     model_meta_data = []
#     layer_type = []
#     for (k, v) in nets[0].state_dict().items():
#         model_meta_data.append(v.shape)
#         layer_type.append(k)

    return users_model, net_glob, initial_state_dict, server_state_dict

print(f'MODEL: {args.model}, Dataset: {args.dataset}')

users_model, net_glob, initial_state_dict, server_state_dict = init_nets(args, dropout_p=0.5)

print(net_glob)

total = 0 
for name, param in net_glob.named_parameters():
    print(name, param.size())
    total += np.prod(param.size())
    #print(np.array(param.data.cpu().numpy().reshape([-1])))
    #print(isinstance(param.data.cpu().numpy(), np.array))
print(total)

################################# Initializing Clients 

clients = []

for idx in range(args.num_users):
    
    dataidxs = net_dataidx_map[idx]
    if net_dataidx_map_test is None:
        dataidx_test = None 
    else:
        dataidxs_test = net_dataidx_map_test[idx]

    #print(f'Initializing Client {idx}')

    noise_level = args.noise
    if idx == args.num_users - 1:
        noise_level = 0

    if args.noise_type == 'space':
        train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, 
                                                                       args.datadir, args.local_bs, 32, 
                                                                       dataidxs, noise_level, idx, 
                                                                       args.num_users-1, 
                                                                       dataidxs_test=dataidxs_test)
    else:
        noise_level = args.noise / (args.num_users - 1) * idx
        train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, 
                                                                       args.datadir, args.local_bs, 32, 
                                                                       dataidxs, noise_level, 
                                                                       dataidxs_test=dataidxs_test)
    
    clients.append(Client_FLIS(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep, 
               args.lr, args.momentum, args.device, train_dl_local, test_dl_local))
    

###################################### Federation 

float_formatter = "{:.4f}".format
#np.set_printoptions(formatter={float: float_formatting_function})
np.set_printoptions(formatter={'float_kind':float_formatter})

loss_train = []

init_tracc_pr = []  # initial train accuracy for each round 
final_tracc_pr = [] # final train accuracy for each round 

init_tacc_pr = []  # initial test accuarcy for each round 
final_tacc_pr = [] # final test accuracy for each round

init_tloss_pr = []  # initial test loss for each round 
final_tloss_pr = [] # final test loss for each round 

clients_best_acc = [0 for _ in range(args.num_users)]
w_locals, loss_locals = [], []

init_local_tacc = []       # initial local test accuracy at each round 
final_local_tacc = []  # final local test accuracy at each round 

init_local_tloss = []      # initial local test loss at each round 
final_local_tloss = []     # final local test loss at each round 

ckp_avg_tacc = []
ckp_avg_best_tacc = []

w_glob_per_cluster = []

users_best_acc = [0 for _ in range(args.num_users)]
best_glob_acc = 0 
best_glob_w = None

idx_cluster = 0 
selected_clusters = {i: [] for i in range(10)}

clust_err = []
clust_acc = []

count_clusters = {i:0 for i in range(1, args.rounds)}
for iteration in range(args.rounds):
        
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    
    print(f'###### ROUND {iteration+1} ######')
    print(f'Clients {idxs_users}')
    
    selected_clusters.clear()
    if iteration+1 > 1:
        selected_clusters = {i: [] for i in range(len(clusters))}
    for idx in idxs_users:
        if iteration+1 > 1:
            assert (len(clusters) == len(w_glob_per_cluster))
            count_clusters[iteration] = len(clusters)
            
            acc_select = []
            for i in range(len(clusters)):
                clients[idx].set_state_dict(copy.deepcopy(w_glob_per_cluster[i])) 

                loss, acc = clients[idx].eval_test() 
                acc_select.append(acc)

            idx_cluster = np.argmax(acc_select)
            clients[idx].set_state_dict(copy.deepcopy(w_glob_per_cluster[idx_cluster])) 

            selected_clusters[idx_cluster].append(idx)
            print(f'Client {idx}, Select Cluster: {idx_cluster}')
            print(f'acc clusters: {acc_select}')

            
        loss, acc = clients[idx].eval_test()        
            
        init_local_tacc.append(acc)
        init_local_tloss.append(loss)
            
        loss = clients[idx].train(is_print=False)
                        
        w_locals.append(copy.deepcopy(clients[idx].get_state_dict()))
        loss_locals.append(copy.deepcopy(loss))
                        
        loss, acc = clients[idx].eval_test()
        
        if acc > clients_best_acc[idx]:
            clients_best_acc[idx] = acc
  
        final_local_tacc.append(acc)
        final_local_tloss.append(loss)           
    
    # Finding clusters 
    clusters, clusters_bm, w_locals_clusters, clients_correct_pred_per_label, clients_similarity, mat_sim, A = \
    cluster_logits(idxs_users, clients, shared_data_loader, args, alpha=args.cluster_alpha, 
                   nclasses=args.nclasses, nsamples=args.nsamples_shared)

    ## Clustering Error 
    c_err, c_acc = error_clustering(clusters_bm, idxs_users, traindata_cls_counts)
    clust_err.append(c_err)
    clust_acc.append(c_acc)
    
    clusters_label = []
    clusters_client_label = []
    for c in clusters: 
        temp = []
        temp2 = []
        for k in c:
            temp2.append(list(traindata_cls_counts[k].keys()))
            temp.extend(list(traindata_cls_counts[k].keys()))
        clusters_client_label.append(temp2)
        temp = list(set(temp))
        clusters_label.append(temp)

    # FedAvg per cluster
    
    total_data_points = [sum([len(net_dataidx_map[r]) for r in clust]) for clust in clusters]
    fed_avg_freqs = [[len(net_dataidx_map[r]) / total_data_points[clust_id] for r in clusters[clust_id]] 
                     for clust_id in range(len(clusters))]
        
    w_glob_per_cluster.clear()
    acc_glob_pc = []
    for i in range(len(clusters)):
        ww = FedAvg(w_locals_clusters[i], weight_avg = fed_avg_freqs[i])
        w_glob_per_cluster.append(ww)
        net_glob.load_state_dict(copy.deepcopy(ww))
        _, acc = eval_test(net_glob, args, test_dl_global)
        if acc > best_glob_acc:
            best_glob_acc = acc 
            best_glob_w = copy.deepcopy(ww)
        acc_glob_pc.append(acc)
        
    idx_cluster = np.argmax(acc_glob_pc)
        
    # update global weights
    w_glob = FedAvg(w_locals)

    # copy weight to net_glob
    net_glob.load_state_dict(w_glob)
    
    # print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    avg_init_tloss = sum(init_local_tloss) / len(init_local_tloss)
    avg_init_tacc = sum(init_local_tacc) / len(init_local_tacc)
    avg_final_tloss = sum(final_local_tloss) / len(final_local_tloss)
    avg_final_tacc = sum(final_local_tacc) / len(final_local_tacc)
         
    print('## END OF ROUND ##')
    template = 'Average Train loss {:.3f}'
    print(template.format(loss_avg))
    
    template = "AVG Init Test Loss: {:.3f}, AVG Init Test Acc: {:.3f}"
    print(template.format(avg_init_tloss, avg_init_tacc))
    
    template = "AVG Final Test Loss: {:.3f}, AVG Final Test Acc: {:.3f}"
    print(template.format(avg_final_tloss, avg_final_tacc))
    
    if iteration%args.print_freq == 0 and iteration != 0:
        print('--- PRINTING ALL CLIENTS STATUS ---')
        current_acc = []
        for k in range(args.num_users):
            loss, acc = clients[k].eval_test() 
            current_acc.append(acc)
            
            template = ("Client {:3d}, labels {}, count {}, best_acc {:3.3f}, current_acc {:3.3f} \n")
            print(template.format(k, traindata_cls_counts[k], clients[k].get_count(),
                                  clients_best_acc[k], current_acc[-1]))
            
        template = ("Round {:1d}, Avg current_acc {:3.3f}, Avg best_acc {:3.3f}")
        print(template.format(iteration+1, np.mean(current_acc), np.mean(clients_best_acc)))
        
        ckp_avg_tacc.append(np.mean(current_acc))
        ckp_avg_best_tacc.append(np.mean(clients_best_acc))
    
    print('----- Analysis End of Round -------')
    for idx in idxs_users:
        print(f'Client {idx}, Count: {clients[idx].get_count()}, Labels: {traindata_cls_counts[idx]}')
    
    print('')
    for idx in idxs_users:
        print(f'Client {idx}, Correct_pred_per_label: {clients_correct_pred_per_label[idx]}')
        #print(f'similarity: {clients_similarity[idx]:}')
        
#     print('')
#     print(f'Similarity Matrix: \n {mat_sim}')
#     print('')
#     print(f'Selected Clusters {selected_clusters}')
#     print('')
#     print(f'New Cluster {clusters}')
#     print(f'Error of Clustering {clust_err[-1]}')
#     print(f'Acc of Clustering {clust_acc[-1]}')
#     print(f'Clusters Lables {clusters_label}')
#     print(f'Clusters Clients Lables {clusters_client_label}')
    print(f'Clusters Glob Acc: {acc_glob_pc}')
    
    loss_train.append(loss_avg)
    
    init_tacc_pr.append(avg_init_tacc)
    init_tloss_pr.append(avg_init_tloss)
    
    final_tacc_pr.append(avg_final_tacc)
    final_tloss_pr.append(avg_final_tloss)
    
    #break;
    ## clear the placeholders for the next round 
    w_locals.clear()
    loss_locals.clear()
    init_local_tacc.clear()
    init_local_tloss.clear()
    final_local_tacc.clear()
    final_local_tloss.clear()
    
    ## calling garbage collector 
    gc.collect()
    
############################### Saving Training Results 
# with open(path+str(args.trial)+'_loss_train.npy', 'wb') as fp:
#     loss_train = np.array(loss_train)
#     np.save(fp, loss_train)
    
# with open(path+str(args.trial)+'_init_tacc_pr.npy', 'wb') as fp:
#     init_tacc_pr = np.array(init_tacc_pr)
#     np.save(fp, init_tacc_pr)
    
# with open(path+str(args.trial)+'_init_tloss_pr.npy', 'wb') as fp:
#     init_tloss_pr = np.array(init_tloss_pr)
#     np.save(fp, init_tloss_pr)
    
# with open(path+str(args.trial)+'_final_tacc_pr.npy', 'wb') as fp:
#     final_tacc_pr = np.array(final_tacc_pr)
#     np.save(fp, final_tacc_pr)
    
# with open(path+str(args.trial)+'_final_tloss_pr.npy', 'wb') as fp:
#     final_tloss_pr = np.array(final_tloss_pr)
#     np.save(fp, final_tloss_pr)
    
# with open(path+str(args.trial)+'_best_glob_w.pt', 'wb') as fp:
#     torch.save(best_glob_w, fp)
############################### Printing Final Test and Train ACC / LOSS
test_loss = []
test_acc = []
train_loss = []
train_acc = []

for idx in range(args.num_users):        
    loss, acc = clients[idx].eval_test()
        
    test_loss.append(loss)
    test_acc.append(acc)
    
    loss, acc = clients[idx].eval_train()
    
    train_loss.append(loss)
    train_acc.append(acc)

test_loss = sum(test_loss) / len(test_loss)
test_acc = sum(test_acc) / len(test_acc)

train_loss = sum(train_loss) / len(train_loss)
train_acc = sum(train_acc) / len(train_acc)

print(f'Train Loss: {train_loss}, Test_loss: {test_loss}')
print(f'Train Acc: {train_acc}, Test Acc: {test_acc}')

print(f'Best Clients AVG Acc: {np.mean(clients_best_acc)}')
print(f'Best Global Model Acc: {best_glob_acc}')

Avg_clusters_per_round = np.mean(list(count_clusters.values()))
total_clusters = np.sum(list(count_clusters.values()))
print(f'Total_clusters: {total_clusters}, Avg clusters per round: {Avg_clusters_per_round}')

avg_clust_err = np.mean(clust_err)
avg_clust_acc = np.mean(clust_acc)
print(f'Avg clustering error: {avg_clust_err}, Avg clustering acc: {avg_clust_acc}')

############################# Saving Print Results 
with open(path+str(args.trial)+'_final_results.txt', 'a') as text_file:
    print(f'Train Loss: {train_loss}, Test_loss: {test_loss}', file=text_file)
    print(f'Train Acc: {train_acc}, Test Acc: {test_acc}', file=text_file)

    print(f'Best Clients AVG Acc: {np.mean(clients_best_acc)}', file=text_file)
    print(f'Best Global Model Acc: {best_glob_acc}', file=text_file)
    print(f'Total_clusters: {total_clusters}, Avg clusters per round: {Avg_clusters_per_round}', file=text_file)
    print(f'Avg clustering error: {avg_clust_err}, Avg clustering acc: {avg_clust_acc}')

    