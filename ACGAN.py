import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import resnet
import preact_resnet


class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, dataset = 'mnist'):
        super(generator, self).__init__()

        self.input_height = 28
        self.input_width = 28
        self.input_dim = 100 + 10
        self.output_dim = 1

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)

        return x


class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset='mnist'):
        super(discriminator, self).__init__()

        self.input_height = 28
        self.input_width = 28
        self.input_dim = 1
        self.output_dim = 1

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.dc = nn.Sequential(
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        x = self.fc1(x)
        d = self.dc(x)

        return d


class lenet(nn.Module):
    def __init__(self, dataset='mnist'):
        super(lenet, self).__init__()

        self.input_height = 28
        self.input_width = 28
        self.input_dim = 1
        self.class_num = 10

        self.conv1 = nn.Conv2d(self.input_dim, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.class_num)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class resnet18(nn.Module):
    def __init__(self, dataset='mnist'):
        super(resnet18, self).__init__()

        self.input_height = 28
        self.input_width = 28
        self.input_dim = 1
        self.class_num = 10

        self.input_shape = (self.input_dim, self.input_height, self.input_width)
        self.model = resnet.ResNet18(self.input_shape, self.class_num)

    def forward(self, x):
        return self.model(x)

class resnet18pa(nn.Module):
    def __init__(self, dataset='mnist'):
        super(resnet18pa, self).__init__()

        self.input_height = 28
        self.input_width = 28
        self.input_dim = 1
        self.class_num = 10

        self.input_shape = (self.input_dim, self.input_height, self.input_width)
        self.model = preact_resnet.PreActResNet18(self.input_shape, self.class_num)

    def forward(self, x):
        return self.model(x)



class ACGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type

        # networks init
        self.G = generator(self.dataset)
        self.D = discriminator(self.dataset)

        # self.C = lenet(self.dataset)
        self.C = resnet18(self.dataset)
        # self.C = resnet18pa(self.dataset)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        # self.C_optimizer = optim.SGD(self.C.parameters(), lr=0.01)
        self.C_optimizer = optim.Adadelta(self.C.parameters()) # , lr=0.1, rho=0.9, eps=1e-8

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.C.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.CE_loss = nn.CrossEntropyLoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        utils.print_network(self.C)
        print('-----------------------------------------------')

        # load mnist
        if self.dataset == 'mnist':
            self.data_X, self.data_Y, self.X_test, self.y_test_vec = utils.load_mnist(args.dataset)

        self.z_dim = 100
        self.y_dim = 10

        # fixed noise & condition
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(10):
            self.sample_z_[i*self.y_dim] = torch.rand(1, self.z_dim)
            for j in range(1, self.y_dim):
                self.sample_z_[i*self.y_dim + j] = self.sample_z_[i*self.y_dim]

        temp = torch.zeros((10, 1))
        for i in range(self.y_dim):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(10):
            temp_y[i*self.y_dim: (i+1)*self.y_dim] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.y_dim))
        self.sample_y_.scatter_(1, temp_y.type(torch.LongTensor), 1)
        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = Variable(self.sample_z_.cuda(), volatile=True), Variable(self.sample_y_.cuda(), volatile=True)
        else:
            self.sample_z_, self.sample_y_ = Variable(self.sample_z_, volatile=True), Variable(self.sample_y_, volatile=True)

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['C_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(torch.zeros(self.batch_size, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))

        self.D.train()
        self.C.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter in range(len(self.data_X) // self.batch_size):
                x_ = self.data_X[iter*self.batch_size:(iter+1)*self.batch_size]
                z_ = torch.rand((self.batch_size, self.z_dim))
                # z_ = torch.Tensor(self.batch_size, self.z_dim).normal_(0, 1)
                y_vec_ = self.data_Y[iter*self.batch_size:(iter+1)*self.batch_size]

                if self.gpu_mode:
                    x_, z_, y_vec_ = Variable(x_.cuda()), Variable(z_.cuda()), Variable(y_vec_.cuda())
                else:
                    x_, z_, y_vec_ = Variable(x_), Variable(z_), Variable(y_vec_)

                # update D network
                self.D_optimizer.zero_grad()
                self.C_optimizer.zero_grad()

                D_real = self.D(x_)
                C_real = self.C(x_)

                D_real_loss = self.BCE_loss(D_real, self.y_real_)
                C_real_loss = self.CE_loss(C_real, torch.max(y_vec_, 1)[1])

                G_ = self.G(z_, y_vec_)
                D_fake = self.D(G_)
                C_fake = self.C(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)
                C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                D_loss = D_real_loss + D_fake_loss
                C_loss = C_real_loss + C_fake_loss

                self.train_hist['D_loss'].append(D_loss.item())
                self.train_hist['C_loss'].append(C_loss.item())

                D_loss.backward(retain_graph=True)
                self.D_optimizer.step()

                C_loss.backward(retain_graph=True)
                self.C_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_, y_vec_)
                D_fake = self.D(G_)
                C_fake = self.C(G_)

                G_loss = self.BCE_loss(D_fake, self.y_real_)
                C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                G_loss += C_fake_loss
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward(retain_graph=True)
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, C_loss: %.8f" %
                          ((epoch + 1), (iter + 1), len(self.data_X) // self.batch_size, D_loss.item(), G_loss.item()
                           , C_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            self.visualize_results((epoch+1), fix=False)

            print('\n[INFO]: Test the classifier:')
            # self.C.eval()
            correct = 0
            nb_test = len(self.X_test)

            for iter in range(nb_test // self.batch_size):
                x_ = self.X_test[iter*self.batch_size:(iter+1)*self.batch_size]
                y_vec_ = self.y_test_vec[iter*self.batch_size:(iter+1)*self.batch_size]

                if self.gpu_mode:
                    x_, y_vec_ = Variable(x_.cuda()), Variable(y_vec_.cuda())
                else:
                    x_, y_vec_ = Variable(x_), Variable(y_vec_)

                outputs = self.C(x_)

                #  C_real_loss = self.CE_loss(C_real, torch.max(y_vec_, 1)[1])

                # loss = self.CE_loss(outputs, torch.max(y_vec_, 1)[1])

                pred = outputs.data.max(1)[1] # get the index of the max log-probability
                pred = pred.eq(torch.max(y_vec_, 1)[1].data).cpu().data.float()
                correct += pred.sum()


            print('Accuracy of the network on the test images: %.2f %%' % (
                100. * correct / nb_test))

            print('[INFO]: Testing finish! \n')


        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_, self.sample_y_)
        else:
            """ random noise """
            temp = torch.LongTensor(self.batch_size, 1).random_() % 10
            sample_y_ = torch.FloatTensor(self.batch_size, 10)
            sample_y_.zero_()
            sample_y_.scatter_(1, temp, 1)
            if self.gpu_mode:
                sample_z_, sample_y_ = Variable(torch.rand((self.batch_size, self.z_dim)).cuda(), volatile=True), \
                                       Variable(sample_y_.cuda(), volatile=True)
            else:
                sample_z_, sample_y_ = Variable(torch.rand((self.batch_size, self.z_dim)), volatile=True), \
                                       Variable(sample_y_, volatile=True)

            samples = self.G(sample_z_, sample_y_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))
        torch.save(self.C.state_dict(), os.path.join(save_dir, self.model_name + '_C.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
        self.C.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_C.pkl')))
