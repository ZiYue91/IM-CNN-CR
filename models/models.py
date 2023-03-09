def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'test':
        from .test_model import TestModel
        model = TestModel()
    elif opt.model == 'unet':
        from .unet_model import UNetModel
        model = UNetModel()
    elif opt.model == 'slopenet':
        from .slopenet_model import SlopeNetModel
        model = SlopeNetModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
