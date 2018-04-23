[train_x, train_t, valid_x, valid_t, test_x, test_t, vocab] = load_data(1);
voc = [];
for i=1:length(vocab)
  voc = [voc;vocab{i}];
end
csvwrite("vc.csv",voc);
csvwrite("trn_dat.csv",squeeze(train_x)');
csvwrite("trn_lab.csv",squeeze(train_t));
csvwrite("tst_dat.csv",test_x');
csvwrite("tst_lab.csv",test_t');
csvwrite("val_dat.csv",valid_x');
csvwrite("val_lab.csv",valid_t');