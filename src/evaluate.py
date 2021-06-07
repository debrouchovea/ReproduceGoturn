from __future__ import absolute_import

from got10k.experiments import ExperimentOTB

from goturn import TrackerGOTURN


if __name__ == '__main__':
    # setup tracker
    # net_path = '/content/drive/My Drive/Goturnreproduce/pytorch_goturn.pth.tar'
    #net_path='/content/drive/My Drive/Goturnreproduce/Checkpoints/model_itr_90000_loss_41.181.pth.tar'
    net_path='/content/drive/My Drive/Goturnreproduce/Checkpoints/mobilenet/model_itr_190000_loss_51.125.pth.tar'
    tracker = TrackerGOTURN(net_path=net_path)

    # setup experiments
    # got10k toolkit expects either extracted directories or zip files for
    # all sequences in OTB data directory
   # experiments = [
   #     ExperimentOTB('../data/OTB', version=2013),
   #     ExperimentOTB('../data/OTB', version=2015)
   # ]
    experiments = [
        ExperimentOTB(root_dir = '/content/drive/My Drive/Goturnreproduce/DataEvaluate/OTB', result_dir = '/content/drive/My Drive/Goturnreproduce/DataEvaluate/Result', report_dir = '/content/drive/My Drive/Goturnreproduce/DataEvaluate/Report', version=2013)]

    # run tracking experiments and report performance
    for e in experiments:
        e.run(tracker, visualize=True)
        e.report([tracker.name])
