import torch
import torch.nn as nn


def eval(args, model, img_encoder, loader, metric, device):
    model.eval()
    img_encoder.eval()
    for i, batch in enumerate(loader):
        # Forward pass
        if args.task == 'twitter_ae':
            aesc_infos = {
                key: value
                for key, value in batch['TWITTER_AE'].items()
            }
        elif args.task == 'twitter_sc':
            aesc_infos = {
                key: value
                for key, value in batch['TWITTER_SC'].items()
            }
        else:
            aesc_infos = {key: value for key, value in batch['AESC'].items()}

        with torch.no_grad():
            imgs_f = [x.numpy().tolist() for x in batch['image_features']]
            imgs_f = torch.tensor(imgs_f).to(device)
            imgs_f, img_mean, img_att = img_encoder(imgs_f)
            img_att = img_att.view(-1, 2048, 49).permute(0, 2, 1)

        predict = model.predict(
            input_ids=batch['input_ids'].to(device),
            # image_features=list(map(lambda x: x.to(device), batch['image_features'])),
            image_features=list(map(lambda x: x.to(device), img_att)),
            attention_mask=batch['attention_mask'].to(device),
            aesc_infos=aesc_infos,
            sentence_length=batch['sentence_ids'].size(-1))

        metric.evaluate(aesc_infos['spans'], predict,
                        aesc_infos['labels'].to(device))
        # break

    res = metric.get_metric()
    model.train()
    return res
