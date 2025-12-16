import torch
import os

#保存最好模型函数
def save_checkpoint(generator, discriminator, epoch, best_loss, checkpoint_dir="./checkpoints/"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 保存生成器和判别器的 state_dict（模型参数）
    torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f"best_generator_model.pth"))

    torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f"best_discriminator_model.pth"))

    data = 'best_loss :'
    epoch = epoch + 1

    with open('./log.txt', 'w', encoding='utf-8') as file :
        file.write(data)
        file.write(str(best_loss))
        file.write('epoch :')
        file.write(str(epoch))
