from dataset import *
from UT_HAR_model import *
from NTU_Fi_model import *
from widar_model import *
from self_supervised_model import *
from MyData_model import *
import torch
from torch.utils.data import DataLoader, ConcatDataset

def load_data_n_model(dataset_name, model_name, root):
    # dummy = np.load(os.path.join(root, 'MyData_HAR/label/y_train.npy'))
    classes = {'UT_HAR_data':7,'NTU-Fi-HumanID':14,'NTU-Fi_HAR':6,'Widar':22,'MyDataset':5}
    if dataset_name == 'UT_HAR_data':
        print('using dataset: UT-HAR DATA')
        data = UT_HAR_dataset(root)
        train_set = torch.utils.data.TensorDataset(data['X_train'],data['y_train'])
        test_set = torch.utils.data.TensorDataset(torch.cat((data['X_val'],data['X_test']),0),torch.cat((data['y_val'],data['y_test']),0))
        train_loader = torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True, drop_last=True) # drop_last=True
        test_loader = torch.utils.data.DataLoader(test_set,batch_size=256,shuffle=False)
        if model_name == 'MLP':
            print("using model: MLP")
            model = UT_HAR_MLP()
            train_epoch = 200
        elif model_name == 'LeNet':
            print("using model: LeNet")
            model = UT_HAR_LeNet()
            train_epoch = 200 #40
        elif model_name == 'ResNet18':
            print("using model: ResNet18")
            model = UT_HAR_ResNet18()
            train_epoch = 200 #70
        elif model_name == 'ResNet50':
            print("using model: ResNet50")
            model = UT_HAR_ResNet50()
            train_epoch = 200 #100
        elif model_name == 'ResNet101':
            print("using model: ResNet101")
            model = UT_HAR_ResNet101()
            train_epoch = 200 #100
        elif model_name == 'RNN':
            print("using model: RNN")
            model = UT_HAR_RNN()
            train_epoch = 3000
        elif model_name == 'GRU':
            print("using model: GRU")
            model = UT_HAR_GRU()
            train_epoch = 200
        elif model_name == 'LSTM':
            print("using model: LSTM")
            model = UT_HAR_LSTM()
            train_epoch = 200
        elif model_name == 'BiLSTM':
            print("using model: BiLSTM")
            model = UT_HAR_BiLSTM()
            train_epoch = 200
        elif model_name == 'CNN+GRU':
            print("using model: CNN+GRU")
            model = UT_HAR_CNN_GRU()
            train_epoch = 200 #20
        elif model_name == 'ViT':
            print("using model: ViT")
            model = UT_HAR_ViT()
            train_epoch = 200 #100
        return train_loader, test_loader, model, train_epoch
    
    
    elif dataset_name == 'NTU-Fi-HumanID':
        print('using dataset: NTU-Fi-HumanID')
        num_classes = classes['NTU-Fi-HumanID']
        train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi-HumanID/test_amp/'), batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi-HumanID/train_amp/'), batch_size=64, shuffle=False)
        if model_name == 'MLP':
            print("using model: MLP")
            model = NTU_Fi_MLP(num_classes)
            train_epoch = 50 #15
        elif model_name == 'LeNet':
            print("using model: LeNet")
            model = NTU_Fi_LeNet(num_classes)
            train_epoch = 50 #20
        elif model_name == 'ResNet18':
            print("using model: ResNet18")
            model = NTU_Fi_ResNet18(num_classes)
            train_epoch = 50 #30
        elif model_name == 'ResNet50':
            print("using model: ResNet50")
            model = NTU_Fi_ResNet50(num_classes)
            train_epoch = 50 #40
        elif model_name == 'ResNet101':
            print("using model: ResNet101")
            model = NTU_Fi_ResNet101(num_classes)
            train_epoch = 50
        elif model_name == 'RNN':
            print("using model: RNN")
            model = NTU_Fi_RNN(num_classes)
            train_epoch = 75
        elif model_name == 'GRU':
            print("using model: GRU")
            model = NTU_Fi_GRU(num_classes)
            train_epoch = 50 #40
        elif model_name == 'LSTM':
            print("using model: LSTM")
            model = NTU_Fi_LSTM(num_classes)
            train_epoch = 50
        elif model_name == 'BiLSTM':
            print("using model: BiLSTM")
            model = NTU_Fi_BiLSTM(num_classes)
            train_epoch = 50
        elif model_name == 'CNN+GRU':
            print("using model: CNN+GRU")
            model = NTU_Fi_CNN_GRU(num_classes)
            train_epoch = 200 #20
        elif model_name == 'ViT':
            print("using model: ViT")
            model = NTU_Fi_ViT(num_classes=num_classes)
            train_epoch = 50
        return train_loader, test_loader, model, train_epoch
    
    
    elif dataset_name == 'NTU-Fi_HAR':
        print('using dataset: NTU-Fi_HAR')
        num_classes = classes['NTU-Fi_HAR']
        train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR/train_amp/'), batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR/test_amp/'), batch_size=64, shuffle=False)
        if model_name == 'MLP':
            print("using model: MLP")
            model = NTU_Fi_MLP(num_classes)
            train_epoch = 30 #10
        elif model_name == 'LeNet':
            print("using model: LeNet")
            model = NTU_Fi_LeNet(num_classes)
            train_epoch = 30 #10
        elif model_name == 'ResNet18':
            print("using model: ResNet18")
            model = NTU_Fi_ResNet18(num_classes)
            train_epoch = 30
        elif model_name == 'ResNet50':
            print("using model: ResNet50")
            model = NTU_Fi_ResNet50(num_classes)
            train_epoch = 30 #40
        elif model_name == 'ResNet101':
            print("using model: ResNet101")
            model = NTU_Fi_ResNet101(num_classes)
            train_epoch = 30
        elif model_name == 'RNN':
            print("using model: RNN")
            model = NTU_Fi_RNN(num_classes)
            train_epoch = 70
        elif model_name == 'GRU':
            print("using model: GRU")
            model = NTU_Fi_GRU(num_classes)
            train_epoch = 30 #20
        elif model_name == 'LSTM':
            print("using model: LSTM")
            model = NTU_Fi_LSTM(num_classes)
            train_epoch = 30 #20
        elif model_name == 'BiLSTM':
            print("using model: BiLSTM")
            model = NTU_Fi_BiLSTM(num_classes)
            train_epoch = 30 #20
        elif model_name == 'CNN+GRU':
            print("using model: CNN+GRU")
            model = NTU_Fi_CNN_GRU(num_classes)
            train_epoch = 100 #20
        elif model_name == 'ViT':
            print("using model: ViT")
            model = NTU_Fi_ViT(num_classes=num_classes)
            train_epoch = 30
        return train_loader, test_loader, model, train_epoch

    elif dataset_name == 'Widar':
        print('using dataset: Widar')
        num_classes = classes['Widar']
        train_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata/train/'), batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata/test/'), batch_size=128, shuffle=False)
        if model_name == 'MLP':
            print("using model: MLP")
            model = Widar_MLP(num_classes)
            train_epoch = 30 #20
        elif model_name == 'LeNet':
            print("using model: LeNet")
            model = Widar_LeNet(num_classes)
            train_epoch = 100 #40
        elif model_name == 'ResNet18':
            print("using model: ResNet18")
            model = Widar_ResNet18(num_classes)
            train_epoch = 100
        elif model_name == 'ResNet50':
            print("using model: ResNet50")
            model = Widar_ResNet50(num_classes)
            train_epoch = 100 #40
        elif model_name == 'ResNet101':
            print("using model: ResNet101")
            model = Widar_ResNet101(num_classes)
            train_epoch = 100
        elif model_name == 'RNN':
            print("using model: RNN")
            model = Widar_RNN(num_classes)
            train_epoch = 500
        elif model_name == 'GRU':
            print("using model: GRU")
            model = Widar_GRU(num_classes)
            train_epoch = 200 
        elif model_name == 'LSTM':
            print("using model: LSTM")
            model = Widar_LSTM(num_classes)
            train_epoch = 200 #20
        elif model_name == 'BiLSTM':
            print("using model: BiLSTM")
            model = Widar_BiLSTM(num_classes)
            train_epoch = 200
        elif model_name == 'CNN+GRU':
            print("using model: CNN+GRU")
            model = Widar_CNN_GRU(num_classes)
            train_epoch = 200 #20
        elif model_name == 'ViT':
            print("using model: ViT")
            model = Widar_ViT(num_classes=num_classes)
            train_epoch = 200
        return train_loader, test_loader, model, train_epoch
    
    # elif dataset_name == 'MyDataset':
    #     print('using dataset: MyDataset')
    #     if model_name in ['LeNet', 'ResNet18', 'ResNet50', 'ResNet101']:
    #         mode = 'cnn'
    #     elif model_name == 'GRU':
    #         mode = 'rnn'
    #     elif model_name == 'CNN+GRU':
    #         mode = 'flat'
    #     else:
    #         raise ValueError(f"Unsupported model: {model_name}")

    #     data_root = os.path.join(root, 'MyData_HAR/data')
    #     label_root = os.path.join(root, 'MyData_HAR/label')

    #     train_set = MyCSIDataset(os.path.join(data_root, 'X_train.npy'), os.path.join(label_root, 'y_train.npy'), mode=mode)
    #     val_set = MyCSIDataset(os.path.join(data_root, 'X_val.npy'), os.path.join(label_root, 'y_val.npy'), mode=mode)
    #     test_set = MyCSIDataset(os.path.join(data_root, 'X_test.npy'), os.path.join(label_root, 'y_test.npy'), mode=mode)

    #     train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    #     test_loader = DataLoader(ConcatDataset([val_set, test_set]), batch_size=128, shuffle=False, num_workers=2)
    #     # data_root = os.path.join(root, 'MyData_HAR/data')
    #     # label_root = os.path.join(root, 'MyData_HAR/label')
    #     # train_set = MyCSIDataset(os.path.join(data_root, 'X_train.npy'), os.path.join(label_root, 'y_train.npy'), mode)
    #     # test_set = MyCSIDataset(
    #     #     np.concatenate([np.load(os.path.join(data_root, 'X_val.npy')), np.load(os.path.join(data_root, 'X_test.npy'))]),
    #     #     np.concatenate([np.load(os.path.join(label_root, 'y_val.npy')), np.load(os.path.join(label_root, 'y_test.npy'))]),
    #     #     mode
    #     # )

    #     # train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    #     # test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    #     num_classes = classes['MyDataset']
    #     if model_name == 'LeNet':
    #         print("using model: LeNet")
    #         model = MyData_LeNet(num_classes)
    #         train_epoch = 30
    #     elif model_name == 'ResNet18':
    #         print("using model: ResNet18")
    #         model = MyData_ResNet18(num_classes)
    #         train_epoch = 30
    #     elif model_name == 'ResNet50':
    #         print("using model: ResNet50")
    #         model = MyData_ResNet50(num_classes)
    #         train_epoch = 30
    #     elif model_name == 'ResNet101':
    #         print("using model: ResNet101")
    #         model = MyData_ResNet101(num_classes)
    #         train_epoch = 30
    #     elif model_name == 'GRU':
    #         print("using model: GRU")
    #         model = MyData_GRU(num_classes)
    #         train_epoch = 30
    #     elif model_name == 'CNN+GRU':
    #         print("using model: CNN+GRU")
    #         model = MyData_CNN_GRU(num_classes)
    #         train_epoch = 30
    #     else:
    #         raise ValueError(f"Unsupported model: {model_name}")
    #     return train_loader, test_loader, model, train_epoch
    
    # elif dataset_name == 'MyDataset':
    #     print('using dataset: MyDataset')

    #     cnn_models = ['LeNet', 'ResNet18', 'ResNet50', 'ResNet101']
    #     # segment_models = ['GRU', 'CNN+GRU', 'SNN']

    #     if model_name in cnn_models:
    #         mode = 'cnn'
    #         prefix = 'frame'
    #     elif model_name == 'GRU':
    #         mode = 'rnn'
    #         prefix = 'segment'
    #     elif model_name in ['CNN+GRU', 'SNN']:
    #         mode = 'snn'
    #         prefix = 'segment'
    #     else:
    #         raise ValueError(f"Unsupported model: {model_name}")

    #     data_root = os.path.join(root, 'MyData_HAR')
    #     data_path = os.path.join(data_root, f'{prefix}_{{}}.npy')
    #     label_path = os.path.join(data_root, f'y_{prefix}_{{}}.npy')

    #     train_set = MyCSIDataset(data_path.format('train'), label_path.format('train'), mode=mode)
    #     val_set = MyCSIDataset(data_path.format('val'), label_path.format('val'), mode=mode)
    #     test_set = MyCSIDataset(data_path.format('test'), label_path.format('test'), mode=mode)

    #     train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    #     test_loader = DataLoader(ConcatDataset([val_set, test_set]), batch_size=128, shuffle=False, num_workers=2)

    #     num_classes = classes['MyDataset']

    #     if model_name == 'LeNet':
    #         print("using model: LeNet")
    #         model = MyData_LeNet(num_classes)
    #         train_epoch = 30
    #     elif model_name == 'ResNet18':
    #         print("using model: ResNet18")
    #         model = MyData_ResNet18(num_classes)
    #         train_epoch = 30
    #     elif model_name == 'ResNet50':
    #         print("using model: ResNet50")
    #         model = MyData_ResNet50(num_classes)
    #         train_epoch = 30
    #     elif model_name == 'ResNet101':
    #         print("using model: ResNet101")
    #         model = MyData_ResNet101(num_classes)
    #         train_epoch = 30
    #     elif model_name == 'GRU':
    #         print("using model: GRU")
    #         model = MyData_GRU(num_classes)
    #         train_epoch = 30
    #     elif model_name == 'CNN+GRU':
    #         print("using model: CNN+GRU")
    #         model = MyData_CNN_GRU(num_classes)
    #         train_epoch = 30
    #     elif model_name == 'SNN':
    #         print("using model: SNN")
    #         model = MyData_SNN(num_classes)
    #         train_epoch = 30
    #     else:
    #         raise ValueError(f"Unsupported model: {model_name}")
    #     return train_loader, test_loader, model, train_epoch

    # elif dataset_name == 'MyDataset':
    #     print('using dataset: MyDataset')

    #     cnn_models = ['LeNet', 'ResNet18', 'ResNet50', 'ResNet101']
    #     segment_models = ['GRU', 'CNN+GRU', 'SNN']

    #     if model_name in cnn_models:
    #         mode = 'cnn'
    #         prefix = 'frame'

    #         data_root = os.path.join(root, 'MyData_HAR')
    #         data_path = os.path.join(data_root, 'data', f'{prefix}_{{}}.npy')
    #         label_path = os.path.join(data_root, 'label', f'y_{prefix}_{{}}.npy')

    #         train_set = MyCSIDataset(data_path.format('train'), label_path.format('train'), mode=mode)
    #         val_set = MyCSIDataset(data_path.format('val'), label_path.format('val'), mode=mode)
    #         test_set = MyCSIDataset(data_path.format('test'), label_path.format('test'), mode=mode)

    #     elif model_name in segment_models:
    #         mode = 'rnn' if model_name == 'GRU' else 'snn'
    #         data_dir = os.path.join(root, 'MyData_HAR', 'OldData')  # Original per-file data

    #         # Tune these parameters:
    #         SEGMENT_SIZE = 300
    #         STEP_SIZE = 150

    #         print(f"🔁 Using on-the-fly segmentation with segment={SEGMENT_SIZE}, stride={STEP_SIZE}, mode={mode}")
    #         train_set = OTFSegmentPerFileDataset(
    #             folder=data_dir,
    #             segment_size=SEGMENT_SIZE,
    #             step_size=STEP_SIZE,
    #             mode=mode
    #         )

    #         # For simplicity, split 80/20 for val/test
    #         total = len(train_set)
    #         val_size = total // 10
    #         test_size = total // 5
    #         train_size = total - val_size - test_size

    #         train_set, val_set, test_set = torch.utils.data.random_split(
    #             train_set, [train_size, val_size, test_size],
    #             generator=torch.Generator().manual_seed(42)
    #         )

    #     else:
    #         raise ValueError(f"Unsupported model: {model_name}")

    #     train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    #     test_loader = DataLoader(ConcatDataset([val_set, test_set]), batch_size=128, shuffle=False, num_workers=2)

    #     num_classes = classes['MyDataset']

    #     # Model selection
    #     if model_name == 'LeNet':
    #         print("using model: LeNet")
    #         model = MyData_LeNet(num_classes)
    #     elif model_name == 'ResNet18':
    #         print("using model: ResNet18")
    #         model = MyData_ResNet18(num_classes)
    #     elif model_name == 'ResNet50':
    #         print("using model: ResNet50")
    #         model = MyData_ResNet50(num_classes)
    #     elif model_name == 'ResNet101':
    #         print("using model: ResNet101")
    #         model = MyData_ResNet101(num_classes)
    #     elif model_name == 'GRU':
    #         print("using model: GRU")
    #         model = MyData_GRU(num_classes)
    #     elif model_name == 'CNN+GRU':
    #         print("using model: CNN+GRU")
    #         model = MyData_CNN_GRU(num_classes)
    #     elif model_name == 'SNN':
    #         print("using model: SNN")
    #         model = MyData_SNN(num_classes)
    #     else:
    #         raise ValueError(f"Unsupported model: {model_name}")

    #     train_epoch = 30
    #     return train_loader, test_loader, model, train_epoch

    elif dataset_name == 'MyDataset':
        print('using dataset: MyDataset')

        data_root = os.path.join(root, 'MyData_HAR/data')
        x_path = os.path.join(data_root, 'x_{}.npy')
        y_path = os.path.join(data_root, 'y_{}.npy')

        train_set = MyCSIDataset(x_path.format('train'), y_path.format('train'))
        val_set   = MyCSIDataset(x_path.format('val'),   y_path.format('val'))
        test_set  = MyCSIDataset(x_path.format('test'),  y_path.format('test'))

        train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
        val_loader   = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2)
        test_loader  = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

        num_classes = classes['MyDataset']

        if model_name == 'LeNet':
            print("using model: LeNet")
            model = MyData_LeNet(num_classes)
        elif model_name == 'GRU':
            print("using model: GRU")
            model = MyData_GRU(num_classes, input_dim=4004, reduced_dim=512, hidden_dim=128)
        # same idea for ResNet, CNN+GRU, etc.

        train_epoch = 30
        return train_loader, val_loader, test_loader, model, train_epoch





def load_unsupervised_data_n_model(model_name,root):
    HAR_train_dataset=CSI_Dataset(root+'NTU-Fi_HAR/train_amp/')
    HAR_test_dataset=CSI_Dataset(root+'NTU-Fi_HAR/test_amp/')
    unsupervised_train_dataset = torch.utils.data.ConcatDataset([HAR_train_dataset,HAR_test_dataset])
    unsupervised_train_loader = torch.utils.data.DataLoader(dataset=unsupervised_train_dataset, batch_size=64, shuffle=True)
    supervised_train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root+'NTU-Fi-HumanID/test_amp/'), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root+'NTU-Fi-HumanID/train_amp/'), batch_size=64, shuffle=False)
    if model_name == 'MLP':
        print("using model: MLP_Parrallel")
        model = MLP_Parrallel()
    elif model_name == 'LeNet':
        print("using model: CNN_Parrallel")
        model = CNN_Parrallel()
    elif model_name == 'ResNet18':
        print("using model: ResNet18_Parrallel")
        model = ResNet18_Parrallel()
    elif model_name == 'ResNet50':
        print("using model: ResNet50_Parralle")
        model = ResNet50_Parrallel()
    elif model_name == 'ResNet101':
        print("using model: ResNet101_Parrallel")
        model = ResNet101_Parrallel()
    elif model_name == 'RNN':
        print("using model: RNN_Parrallel")
        model = RNN_Parrallel()
    elif model_name == 'GRU':
        print("using model: GRU_Parrallel")
        model = GRU_Parrallel()
    elif model_name == 'LSTM':
        print("using model: LSTM_Parrallel")
        model = LSTM_Parrallel()
    elif model_name == 'BiLSTM':
        print("using model: BiLSTM_Parrallel")
        model = BiLSTM_Parrallel()
    elif model_name == 'CNN+GRU':
        print("using model: CNN_GRU_Parrallel")
        model = CNN_GRU_Parrallel()
    elif model_name == 'ViT':
        print("using model: ViT_Parrallel")
        model = ViT_Parrallel()
    return unsupervised_train_loader, supervised_train_loader, test_loader, model