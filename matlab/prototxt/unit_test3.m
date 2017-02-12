
r = builder.layers.add_layer('dummy', 'data', {}, 'data');
%r1 = builder.structures.Inception.add_InceptionA('5a', 'data', [96, 96, 64, 96, 64, 96, 96]);
%r1 = builder.structures.Inception.add_InceptionB('5a', 'data', [128, 384, 192, 224, 256, 192, 192, 224, 224, 256]);
%r1 = builder.structures.Inception.add_InceptionC('5a', 'data', [256, 256, 384, 256, 256, 384, 448, 512, 256, 256]);
%r1 = builder.structures.Inception.add_ReductionA('5a', 'data', [1, 2, 3, 4]);
r1 = builder.structures.Inception.add_ReductionB('5a', 'data', [192, 192, 256, 256, 320, 320]);
save_net([r, r1], 'test.prototxt')