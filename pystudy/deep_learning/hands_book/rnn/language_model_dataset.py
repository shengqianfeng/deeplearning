#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : language_model_dataset.py
@Author : jeffsheng
@Date : 2020/1/4 0004
@Desc :  介绍如何预处理一个语言模型数据集（周杰伦专辑歌词）
"""

import tensorflow as tf
import random
import zipfile
import numpy as np

with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
# 前40个字符
"""
想要有直升机
想要和你飞到宇宙去
想要和你融化在一起
融化在宇宙里
我每天每天每
"""
print(corpus_chars[:40])
print("-----------------------------")
# 了打印方便，把换行符替换成空格，然后仅使用前1万个字符来训练模型
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:10000]
"""
想要有直升机 想要和你飞到宇宙去 想要和你融化在一起 融化在宇宙里 我每天每天每天在想想想想著你 这样的甜蜜 让我开始乡相信命运 感谢地心引力 让我碰到你 漂亮的让我面红的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 漂亮的让我面红的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 想要有直升机 想要和你飞到宇宙去 想要和你融化在一起 融化在宇宙里 我每天每天每天在想想想想著你 这样的甜蜜 让我开始乡相信命运 感谢地心引力 让我碰到你 漂亮的让我面红的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 漂亮的让我面红的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 漂亮的让我面红的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 漂亮的让我面红的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 如果说怀疑 可以造句如果说分离 能够翻译 如果这一切 真的可以 我想要将我的寂寞封闭 然后在这里 不限日期 然后将过去 慢慢温习 让我爱上你 那场悲剧 是你完美演出的一场戏 宁愿心碎哭泣 再狠狠忘记 你爱过我的证据 让晶莹的泪滴 闪烁成回忆 伤人的美丽 你的完美主义 太彻底 让我连恨都难以下笔 将真心抽离写成日记 像是一场默剧 你的完美主义 太彻底 分手的话像语言暴力 我已无能为力再提起 决定中断熟悉 然后在这里 不限日期 然后将过去 慢慢温习 让我爱上你 那场悲剧 是你完美演出的一场戏 宁愿心碎哭泣 再狠狠忘记 你爱过我的证据 让晶莹的泪滴 闪烁成回忆 伤人的美丽 你的完美主义 太彻底 让我连恨都难以下笔 将真心抽离写成日记 像是一场默剧 你的完美主义 太彻底 分手的话像语言暴力 我已无能为力再提起 决定中断熟悉 周杰伦 周杰伦 一步两步三步四步望著天 看星星 一颗两颗三颗四颗 连成线一步两步三步四步望著天 看星星 一颗两颗三颗四颗 连成线乘著风 游荡在蓝天边 一片云掉落在我面前 捏成你的形状 随风跟著我 一口一口吃掉忧愁 载著你 彷彿载著阳光 不管到哪里都是晴天 蝴蝶自在飞 花也布满天 一朵一朵因你而香 试图让夕阳飞翔 带领你我环绕大自然 迎著风 开始共渡每一天 手牵手 一步两步三步四步望著天 看星星 一颗两颗三颗四颗 连成线背著背默默许下心愿 看远方的星是否听的见 手牵手 一步两步三步四步望著天 看星星 一颗两颗三颗四颗 连成线背著背默默许下心愿 看远方的星如果听的见 它一定实现它一定实现 载著你 彷彿载著阳光 不管到哪里都是晴天 蝴蝶自在飞 花也布满天 一朵一朵因你而香 试图让夕阳飞翔 带领你我环绕大自然 迎著风 开始共渡每一天 手牵手 一步两步三步四步望著天 看星星 一颗两颗三颗四颗 连成线背著背默默许下心愿 看远方的星是否听的见 手牵手 一步两步三步四步望著天 看星星 一颗两颗三颗四颗 连成线背著背默默许下心愿 看远方的星如果听的见 它一定实现 它一定实现 娘子 娘子却依旧每日 折一枝杨柳 你在那里 在小村外的溪边河口默默等著我 娘子依旧每日折一枝杨柳 你在那里 在小村外的溪边 默默等待 娘子 一壶好酒 再来一碗热粥 配上几斤的牛肉 我说店小二 三两银够不够 景色入秋 漫天黄沙凉过 塞北的客栈人多 牧草有没有 我马儿有些瘦 世事看透 江湖上潮起潮落 什么恩怨过错 在多年以后 还是让人难过 心伤透 娘子她人在江南等我 泪不休 语沉默 娘子却依旧每日折一枝杨柳 在小村外的溪边河口 默默的在等著我 家乡的爹娘早已苍老了轮廓 娘子我欠你太多 一壶好酒 再来一碗热粥 配上几斤的牛肉 我说店小二 三两银够不够 景色入秋 漫天黄沙凉过 塞北的客栈人多 牧草有没有 我马儿有些瘦 天涯尽头 满脸风霜落寞 近乡情怯的我 相思寄红豆 相思寄红豆无能为力的在人海中漂泊心伤透 娘子她人在江南等我 泪不休 语沉默娘子她人在江南等我 泪不休 语沉默 一壶好酒 再来一碗热粥 配上几斤的牛肉 我说店小二 三两银够不够 景色入秋 漫天黄沙凉过 塞北的客栈人多 牧草有没有 我马儿有些瘦 世事看透 江湖上潮起潮落 什么恩怨过错 在多年以后 还是让人难过 心伤透 娘子她人在江南等我 泪不休 语沉默 娘子却依旧每日折一枝杨柳 在小村外的溪边河口 默默的在等著我 家乡的爹娘早已苍老了轮廓 娘子我欠你太多 一壶好酒 再来一碗热粥 配上几斤的牛肉 我说店小二 三两银够不够 景色入秋 漫天黄沙凉过 塞北的客栈人多 牧草有没有 我马儿有些瘦 天涯尽头 满脸风霜落寞 近乡情怯的我 相思寄红豆 相思寄红豆无能为力的在人海中漂泊心伤透 娘子她人在江南等我 泪不休 语沉默 娘子她人在江南等我 泪不休 语沉默娘子 娘子却依旧每日 折一枝杨柳 你在那里 在小村外的溪边河口默默等著我 娘子依旧每日折一枝杨柳 你在那里 在小村外的溪边 默默等待 娘子 娘子 娘子却依旧每日 折一枝杨柳 你在那里 在小村外的溪边河口默默等著我 娘子依旧每日折一枝杨柳 你在那里 在小村外的溪边 默默等待 娘子 有什么不妥 有话就直说 别窝在角落 不爽就反驳 到底拽什么 懂不懂篮球 有种不要走 三对三斗牛 有什么不妥 有话就直说 别窝在角落 不爽就反驳 到底拽什么 懂不懂篮球 有种不要走 三对三斗牛 三分球 它在空中停留 所有人看着我 抛物线进球 单手过人运球 篮下妙传出手 漂亮的假动作 帅呆了我 全场盯人防守 篮下禁区游走 快攻抢篮板球 得分都靠我 你拿着球不投 又不会掩护我 选你这种队友 瞎透了我 说你说 分数怎么停留 一直在停留 谁让它停留的 为什么我女朋友场外加油 你却还让我出糗 你说啊 你怎么抄我球 你说啊 你怎么打我手 你说啊 是不是你不想活 说你怎么面对我 甩开球我满腔的怒火 我想揍你已经很久 别想躲 说你眼睛看着我 别发抖 快给我抬起头 有话去对医药箱说 别怪我 别怪我 说你怎么面对我 甩开球我满腔的怒火 我想揍你已经很久 别想躲 说你眼睛看着我 别发抖 快给我抬起头 有话去对医药箱说 别怪我 别怪我 三分球 它在空中停留 所有人看着我 抛物线进球 单手过人运球 篮下妙传出手 漂亮的假动作 帅呆了我 全场盯人防守 篮下禁区游走 快攻抢篮板球 得分都靠我 你拿着球不投 又不会掩护我 选你这种队友 瞎透了我 说你说 分数怎么停留 一直在停留 谁让它停留的 为什么我女朋友场外加油 你却还让我出糗 你说啊 你怎么抄我球 你说啊 你怎么打我手 你说啊 是不是你不想活 说你怎么面对我 甩开球我满腔的怒火 我想揍你已经很久 别想躲 说你眼睛看着我 别发抖 快给我抬起头 有话去对医药箱说 别怪我 别怪我 说你怎么面对我 甩开球我满腔的怒火 我想揍你已经很久 别想躲 说你眼睛看着我 别发抖 快给我抬起头 有话去对医药箱说 别怪我 别怪我 难过 是因为闷了很久 是因为想了太多 是心理起了作用 你说 苦笑常常陪着你 在一起有点勉强 该不该现在休了我 不想太多 我想一定是我听错弄错搞错 拜托 我想是你的脑袋有问题 随便说说 其实我早已经猜透看透不想多说 只是我怕眼泪撑不住 不懂 你的黑色幽默 想通 却又再考倒我 说散 你想很久了吧? 我不想拆穿你 当作 是你开的玩笑 想通 却又再考倒我 说散 你想很久了吧? 败给你的黑色幽默 不想太多 我想一定是我听错弄错搞错 拜托 我想是你的脑袋有问题 随便说说 其实我早已经猜透看透不想多说 只是我怕眼泪撑不住 不懂 你的黑色幽默 想通 却又再考倒我 说散 你想很久了吧? 我不想拆穿你 当作 是你开的玩笑 想通 却又再考倒我 说散 你想很久了吧? 败给你的黑色幽默 说散 你想很久了吧? 我的认真败给黑色幽默 走过了很多地方 我来到伊斯坦堡 就像是童话故事  有教堂有城堡 每天忙碌地的寻找 到底什么我想要 却发现迷了路怎么找也找不着 心血来潮起个大早 怎么我也睡不着  昨晚梦里你来找 我才  原来我只想要你 陪我去吃汉堡  说穿了其实我的愿望就怎么小 就怎么每天祈祷我的心跳你知道  杵在伊斯坦堡 却只想你和汉堡 我想要你的微笑每天都能看到  我知道这里很美但家乡的你更美走过了很多地方 我来到伊斯坦堡 就像是童话故事 有教堂有城堡 每天忙碌地的寻找 到底什么我想要 却发现迷了路怎么找也找不着 心血来潮起个大早 怎么我也睡不着  昨晚梦里你来找 我才  原来我只想要你 陪我去吃汉堡  说穿了其实我的愿望就怎么小 就怎么每天祈祷我的心跳你知道  杵在伊斯坦堡 却只想你和汉堡 我想要你的微笑每天都能看到  我知道这里很美但家乡的你更美原来我只想要你 陪我去吃汉堡  说穿了其实我的愿望就怎么小 就怎么每天祈祷我的心跳你知道  杵在伊斯坦堡 却只想你和汉堡 我想要你的微笑每天都能看到  我知道这里很美但家乡的你更美 沙漠之中怎么会有泥鳅 话说完飞过一只海鸥 大峡谷的风呼啸而过 是谁说没有 有一条热昏头的响尾蛇 无力的躺在干枯的河 在等待雨季来临变沼泽 灰狼啃食著水鹿的骨头 秃鹰盘旋死盯着腐肉 草原上两只敌对野牛在远方决斗 在一处被废弃的白蛦丘 站着一只饿昏的老斑鸠 印地安老斑鸠 腿短毛不多 几天都没有喝水也能活 脑袋瓜有一点秀逗 猎物死了它比谁都难过 印地安斑鸠 会学人开口 仙人掌怕羞 蜥蝪横著走 这里什么奇怪的事都有 包括像猫的狗 印地安老斑鸠 平常话不多 除非是乌鸦抢了它的窝 它在灌木丛旁邂逅 一只令它心仪的母斑鸠 牛仔红蕃 在小镇 背对背决斗 一只灰狼 问候我 谁是神枪手 巫师 他念念 有词的 对酋长下诅咒 还我骷髅头 这故事 告诉我 印地安的传说 还真是 瞎透了 什么都有 这故事 告诉我 印地安的传说 还真是 瞎透了 什么都有 沙漠之中怎么会有泥鳅 话说完飞过一只海鸥 大峡谷的风呼啸而过 是谁说没有 有一条热昏头的响尾蛇 无力的躺在干枯的河 在等待雨季来临变沼泽 灰狼啃食著水鹿的骨头 秃鹰盘旋死盯着腐肉 草原上两只敌对野牛在远方决斗 在一处被废弃的白蛦丘 站着一只饿昏的老斑鸠 印地安老斑鸠 腿短毛不多 几天都没有喝水也能活 脑袋瓜有一点秀逗 猎物死了它比谁都难过 印地安斑鸠 会学人开口 仙人掌怕羞 蜥蝪横著走 这里什么奇怪的事都有 包括像猫的狗 印地安老斑鸠 平常话不多 除非是乌鸦抢了它的窝 它在灌木丛旁邂逅 一只令它心仪的母斑鸠 印地安老斑鸠 腿短毛不多 几天都没有喝水也能活 脑袋瓜有一点秀逗 猎物死了它比谁都难过 印地安斑鸠 会学人开口 仙人掌怕羞 蜥蝪横著走 这里什么奇怪的事都有 包括像猫的狗 印地安老斑鸠 平常话不多 除非是乌鸦抢了它的窝 它在灌木丛旁邂逅 一只令它心仪的母斑鸠 爱像一阵风 吹完它就走 这样的节奏 谁都无可奈何 没有你以后 我灵魂失控 黑云在降落 我被它拖着走 静静悄悄默默离开 陷入了危险边缘Baby  我的世界已狂风暴雨 Wu  爱情来的太快就像龙卷风 离不开暴风圈来不及逃 我不能再想 我不能再想 我不 我不 我不能 爱情走的太快就像龙卷风 不能承受我已无处可躲 我不要再想 我不要再想 我不 我不 我不要再想你 不知不觉 你已经离开我 不知不觉 我跟了这节奏 后知后觉 又过了一个秋 后知后觉 我该好好生活 我该好好生活 静静悄悄默默离开 陷入了危险边缘Baby  我的世界已狂风暴雨 Wu  爱情来的太快就像龙卷风 离不开暴风圈来不及逃 我不能再想 我不能再想 我不 我不 我不能 爱情走的太快就像龙卷风 不能承受我已无处可躲 我不要再想 我不要再想 我不 我不 我不要再想你 爱情来的太快就像龙卷风 离不开暴风圈来不及逃 我不能再想 我不能再想 我不 我不 我不能 爱情走的太快就像龙卷风 不能承受我已无处可躲 我不要再想 我不要再想 我不 我不 我不要再想你 不知不觉 你已经离开我 不知不觉 我跟了这节奏 后知后觉 又过了一个秋 后知后觉 我该好好生活 我该好好生活 不知不觉 你已经离开我 不知不觉 我跟了这节奏 后知后觉 又过了一个秋 后知后觉 我该好好生活 我该好好生活 不知不觉 你已经离开我 不知不觉 我跟了这节奏 后知后觉 后知后觉 迷迷蒙蒙 你给的梦 出现裂缝 隐隐作痛 怎么沟通 你都没空 说我不懂 说了没用 他的笑容 有何不同 在你心中 我不再受宠 我的天空 是雨是风 还是彩虹 你在操纵 恨自己真的没用 情绪激动 一颗心到现在还在抽痛 还为分手前那句抱歉 在感动 穿梭时间的画面的钟 从反方向开始移动 回到当初爱你的时空 停格内容不忠 所有回忆对着我进攻 我的伤口被你拆封 誓言太沉重泪被纵容 脸上汹涌失控 城市霓虹 不安跳动 染红夜空 过去种种 象一场梦 不敢去碰 一想就痛 心碎内容 每一秒钟 都有不同 你不懂 连一句珍重 也有苦衷 也不想送 寒风中 废墟烟囱 停止转动 一切落空 在人海中 盲目跟从 别人的梦 全面放纵 恨没有用 疗伤止痛 不再感动 没有梦 痛不知轻重 泪水鲜红 全面放纵 恨自己真的没用 情绪激动 一颗心到现在还在抽痛 还为分手前那句抱歉 在感动 穿梭时间的画面的钟 从反方向开始移动 回到当初爱你的时空 停格内容不忠 所有回忆对着我进攻 我的伤口被你拆封 誓言太沉重泪被纵容 脸上汹涌失控 穿梭时间的画面的钟 从反方向开始移动 回到当初爱你的时空 停格内容不忠 所有回忆对着我进攻       所有回忆对着我进攻       古巴比伦王颁布了汉摩拉比法典 刻在黑色的玄武岩 距今已经三千七百多年 你在橱窗前 凝视碑文的字眼 我却在旁静静欣赏你那张我深爱的脸 祭司 神殿 征战 弓箭 是谁的从前 喜欢在人潮中你只属于我的那画面 经过苏美女神身边 我以女神之名许愿 思念像底格里斯河般的漫延 当古文明只剩下难解的语言 传说就成了永垂不朽的诗篇 我给你的爱写在西元前 深埋在美索不达米亚平原 几十个世纪后出土发现 泥板上的字迹依然清晰可见 我给你的爱写在西元前 深埋在美索不达米亚平原 用楔形文字刻下了永远 那已风化千年的誓言 一切又重演 祭司 神殿 征战 弓箭 是谁的从前 喜欢在人潮中你只属于我的那画面 经过苏美女神身边 我以女神之名许愿 思念像底格里斯河般的漫延 当古文明只剩下难解的语言 传说就成了永垂不朽的诗篇 我给你的爱写在西元前 深埋在美索不达米亚平原 几十个世纪后出土发现 泥板上的字迹依然清晰可见 我给你的爱写在西元前 深埋在美索不达米亚平原 用楔形文字刻下了永远 那已风化千年的誓言 一切又重演 我感到很疲倦离家乡还是很远 害怕再也不能回到你身边 我给你的爱写在西元前 深埋在美索不达米亚平原 几十个世纪后出土发现 泥板上的字迹依然清晰可见 我给你的爱写在西元前 深埋在美索不达米亚平原 用楔形文字刻下了永远 那已风化千年的誓言 一切又重演 爱在西元前 爱在西元前 周杰伦   爸我回来了 我听说通常在战争后就会换来和平 为什么看到我的爸爸一直打我妈妈 就因为喝醉酒 他就能拿我妈出气 我真的看不下去 以为我较细汉 从小到大只有妈妈的温暖  为什么我爸爸 那么凶 如果真的我有一双翅膀 二双翅膀 随时出发 偷偷出发 我一定带我妈走  从前的教育别人的家庭  别人的爸爸种种的暴力因素一定都会有原因 但是呢 妈跟我都没有错亏我叫你一声爸  爸我回来了 不要再这样打我妈妈 我说的话你甘会听 不要再这样打我妈妈 难道你手不会痛吗 其实我回家就想要阻止一切 让家庭回到过去甜甜 温馨的欢乐香味 虽然这是我编造出来的事实 有点讽刺 有点酸性  但它确在这快乐社会发生产生共鸣 产生共鸣来阻止一切暴力  眼泪随着音符吸入血液情绪 从小到大你叫我学习你把你当榜样  好多的假像 妈妈常说乖听你爸的话  你叫我怎么跟你像 不要再这样打我妈妈 我说的话 你甘会听 不要再这样打我妈妈 难道你手不会痛吗 不要再这样打我妈妈 难道你手不会痛吗 我叫你爸 你打我妈 这样对吗干嘛这样 何必让酒牵鼻子走 瞎 说都说不听听 痛是我们在痛痛 周杰伦   简单爱 说不上为什么 我变得很主动 若爱上一个人 什么都会值得去做 我想大声宣布 对你依依不舍 连隔壁邻居都猜到我现在的感受 河边的风 在吹着头发飘动 牵着你的手 一阵莫名感动 我想带你 回我的外婆家 一起看着日落 一直到我们都睡着 我想就这样牵着你的手不放开 爱能不能够永远单纯没有悲哀 我 想带你骑单车 我 想和你看棒球 想这样没担忧 唱着歌 一直走 我想就这样牵着你的手不放开 爱可不可以简简单单没有伤害 你 靠着我的肩膀 你 在我胸口睡著 像这样的生活 我爱你 你爱我 我想大声宣布 对你依依不舍 连隔壁邻居都猜到我现在的感受 河边的风 在吹着头发飘动 牵着你的手 一阵莫名感动 我想带你 回我的外婆家 一起看着日落 一直到我们都睡着 我想就这样牵着你的手不放开 爱能不能够永远单纯没有悲哀 我 想带你骑单车 我 想和你看棒球 想这样没担忧 唱着歌 一直走 我想就这样牵着你的手不放开 爱可不可以简简单单没有伤害 你 靠着我的肩膀 你 在我胸口睡著 像这样的生活 我爱你 你爱我 我想就这样牵着你的手不放开 爱能不能够永远单纯没有悲哀 我 想带你骑单车 我 想和你看棒球 想这样没担忧 唱着歌 一直走 我想就这样牵着你的手不放开 爱可不可以简简单单没有伤害 你 靠着我的肩膀 你 在我胸口睡著 像这样的生活 我爱你 你爱我 开不了口 周杰伦 才离开没多久就开始 担心今天的你过得好不好 整个画面是你 想你想的睡不著 嘴嘟嘟那可爱的模样 还有在你身上香香的味道 我的快乐是你 想你想的都会笑 没有你在 我有多难熬  没有你在我有多难熬多烦恼  没有你烦 我有多烦恼  没有你烦我有多烦恼多难熬  穿过云层 我试著努力向你奔跑 爱才送到 你却已在别人怀抱 就是开不了口让她知道 我一定会呵护著你 也逗你笑 你对我有多重要 我后悔没让你知道 安静的听你撒娇 看你睡著一直到老 就是开不了口让她知道 就是那么简单几句 我办不到 整颗心悬在半空 我只能够远远看著 这些我都做得到 但那个人已经不是我 没有你在 我有多难熬  没有你在我有多难熬多烦恼  没有你烦 我有多烦恼  没有你烦我有多烦恼多难熬  穿过云层 我试著努力向你奔跑 爱才送到 你却已在别人怀抱 就是开不了口让她知道 我一定会呵护著你 也逗你笑 你对我有多重要 我后悔没让你知道 安静的听你撒娇 看你睡著一直到老 就是开不了口让她知道 就是那么简单几句 我办不到 整颗心悬在半空 我只能够远远看著 这些我都做得到 但那个人已经不是我 上海一九四三 泛黄的春联还残留在墙上 依稀可见几个字岁岁平安 在我没回去过的老家米缸 爷爷用楷书写一个满 黄金葛爬满了雕花的门窗 夕阳斜斜映在斑驳的砖墙 铺著榉木板的屋内还弥漫 姥姥当年酿的豆瓣酱 我对著黑白照片开始想像 爸和妈当年的模样 说著一口吴侬软语的姑娘缓缓走过外滩 消失的 旧时光 一九四三 在回忆 的路上 时间变好慢 老街坊 小弄堂 是属于那年代白墙黑瓦的淡淡的忧伤 消失的 旧时光 一九四三 回头看 的片段 有一些风霜 老唱盘 旧皮箱 装满了明信片的铁盒里藏著一片玫瑰花瓣 黄金葛爬满了雕花的门窗 夕阳斜斜映在斑驳的砖墙 铺著榉木板的屋内还弥漫 姥姥当年酿的豆瓣酱 我对著黑白照片开始想像 爸和妈当年的模样 说著一口吴侬软语的姑娘缓缓走过外滩 消失的 旧时光 一九四三 在回忆 的路上 时间变好慢 老街坊 小弄堂 是属于那年代白墙黑瓦的淡淡的忧伤 消失的 旧时光 一九四三 回头看 的片段 有一些风霜 老唱盘 旧皮箱 装满了明信片的铁盒里藏著一片玫瑰花瓣 对不起 广场一枚铜币 悲伤得很隐密 它在许愿池里轻轻叹息 太多的我爱你 让它喘不过气 已经 失去意义 戒指在哭泣 静静躺在抽屉 它所拥有的只剩下回忆 相爱还有别离 像无法被安排的雨 随时准备来袭 我怀念起国小的课桌椅 怀念著用铅笔写日记   纪录那最原始的美丽 纪录第一次遇见的你 如果我遇见你是一场悲剧 我想我这辈子注定一个人演戏 最后再一个人慢慢的回忆 没有了过去 我将往事抽离 如果我遇见你是一场悲剧 我可以让生命就这样毫无意义 或许在最后能听到你一句 轻轻的叹息  后悔着对不起 一枚铜币 悲伤得很隐密 它在许愿池里轻轻叹息 太多的我爱你 让它喘不过气 已经 失去意义 戒指在哭泣 静静躺在抽屉 它所拥有的只剩下回忆 相爱还有别离 像无法被安排的雨 随时准备来袭 我怀念起国小的课桌椅 用铅笔写日记 纪录那最原始的美丽 纪录第一次遇见的你 Jay Chou  如果我遇见你是一场悲剧 我想我这辈子注定一个人演戏 最后再一个人慢慢的回忆 没有了过去 我将往事抽离 如果我遇见你是一场悲剧 我可以让生命就这样毫无意义 或许在最后能听到你一句 轻轻的叹息  后悔着对不起 如果我遇见你是一场悲剧 我想我这辈子注定一个人演戏 最后再一个人慢慢的回忆 没有了过去 我将往事抽离 如果我遇见你是一场悲剧 我轻轻的叹息 后悔着对不起 藤蔓植物 爬满了伯爵的坟墓 古堡里一片荒芜 长满杂草的泥土 不会骑扫把的胖女巫 用拉丁文念咒语啦啦呜 她养的黑猫笑起来像哭 啦啦啦呜 用水晶球替人占卜 她说下午三点阳光射进教堂的角度 能知道你前世是狼人还是蝙蝠 古堡主人威廉二世满脸的落腮胡 习惯 在吸完血后开始打呼 管家是一只会说法语举止优雅的猪 吸血前会念约翰福音做为弥补 拥有一双蓝色眼睛的凯萨琳公主 专吃 有AB血型的公老鼠 恍恍惚惚 是谁的脚步 银制茶壶 装蟑螂蜘蛛 辛辛苦苦 全家怕日出 白色蜡烛 温暖了空屋 白色蜡烛 温暖了空屋藤蔓植物 爬满了伯爵的坟墓 古堡里一片荒芜 长满杂草的泥土 不会骑扫把的胖女巫 用拉丁文念咒语啦啦呜 她养的黑猫笑起来像哭 啦啦啦呜 用水晶球替人占卜 她说下午三点阳光射进教堂的角度 能知道你前世是狼人还是蝙蝠 古堡主人威廉二世满脸的落腮胡 习惯 在吸完血后开始打呼 管家是一只会说法语举止优雅的猪 吸血前会念约翰福音做为弥补 拥有一双蓝色眼睛的凯萨琳公主 专吃 有AB血型的公老鼠 恍恍惚惚 是谁的脚步 银制茶壶 装蟑螂蜘蛛 辛辛苦苦 全家怕日出 白色蜡烛 温暖了空屋 白色蜡烛 温暖了空屋 双截棍 岩烧店的烟味弥漫 隔壁是国术馆 店里面的妈妈桑 茶道 有三段 教拳脚武术的老板 练铁沙掌 耍杨家枪 硬底子功夫最擅长 还会金钟罩铁步衫 他们儿子我习惯 从小就耳濡目染 什么刀枪跟棍棒 我都耍的有模有样 什么兵器最喜欢 双截棍柔中带刚 想要去河南嵩山 学少林跟武当 干什么 干什么 呼吸吐纳心自在 干什么 干什么 气沉丹田手心开 干什么 干什么 日行千里系沙袋 飞檐走壁莫奇怪 去去就来 一个马步向前 一记左钩拳 右钩拳 一句惹毛我的人有危险 一再重演 一根我不抽的菸 一放好多年 它一直在身边 干什么 干什么 我打开任督二脉 干什么 干什么 东亚病夫的招牌 干什么 干什么 已被我一脚踢开 快使用双截棍 哼哼哈兮 快使用双截棍 哼哼哈兮 习武之人切记 仁者无敌 是谁在练太极 风生水起 快使用双截棍 哼哼哈兮 快使用双截棍 哼哼哈兮 如果我有轻功 飞檐走壁 为人耿直不屈 一身正气 他们儿子我习惯 从小就耳濡目染 什么刀枪跟棍棒 我都耍的有模有样 什么兵器最喜欢 双截棍柔中带刚 想要去河南嵩山 学少林跟武当 快使用双截棍 哼哼哈兮 快使用双截棍 哼哼哈兮 习武之人切记 仁者无敌 是谁在练太极 风生水起 快使用双截棍 哼哼哈兮 快使用双截棍 哼哼哈兮 如果我有轻功 飞檐走壁 为人耿直不屈 一身正气 他们儿子我习惯 从小就耳濡目染 什么刀枪跟棍棒 我都耍的有模有样 什么兵器最喜欢 双截棍柔中带刚 想要去河南嵩山 学少林跟武当 快使用双截棍 哼哼哈兮 快使用双截棍 哼哼哈兮 习武之人切记 仁者无敌 是谁在练太极 风生水起 快使用双截棍 哼哼哈兮 快使用双截棍 哼哼哈兮 如果我有轻功 飞檐走壁 为人耿直不屈 一身正气 快使用双截棍 哼 我用手刀防御 哼 漂亮的回旋踢 周杰伦 安静 只剩下钢琴陪
"""
print(corpus_chars)

print("---------建立字符索引----------")
"""
1 将每个字符映射成一个从0开始的连续整数，又称索引，来方便之后的数据处理
2 为了得到索引，我们将数据集里所有不同字符取出来，然后将其逐一映射到索引来构造词典
3 接着，打印vocab_size，即词典中不同字符的个数，又称词典大小
"""
idx_to_char = list(set(corpus_chars))   # set() 函数创建一个无序不重复元素集
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)]) # 建立key为字符，value为索引的字典
vocab_size = len(char_to_idx)
print(vocab_size) # 1027

# 训练数据集中每个字符转化为索引，并打印前20个字符及其对应的索引
corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:20]
# chars: 想要有直升机 想要和你飞到宇宙去 想要和
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
# indices: [962, 167, 500, 1014, 791, 146, 1004, 962, 167, 11, 370, 880, 122, 387, 912, 171, 1004, 962, 167, 11]
print('indices:', sample)

print("------------时序数据的采样------------")
"""
在训练中我们需要每次随机读取小批量样本和标签
1 时序数据的一个样本通常包含连续的字符。假设时间步数为5，样本序列为5个字符，即“想”“要”“有”“直”“升”。
2 该样本的标签序列为这些字符分别在训练集中的下一个字符，即“要”“有”“直”“升”“机”。
3 两种方式对时序数据进行采样，分别是随机采样和相邻采样
"""

"""
随机采样：
1 在随机采样中，每个样本是原始序列上任意截取的一段序列
2 相邻的两个随机小批量在原始序列上的位置不一定相毗邻，因此，不能用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态。
3 在训练模型时，每次随机采样前都需要重新初始化隐藏状态
"""
# 本函数已保存在d2lzh_tensorflow2包中方便以后使用
def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    """
    :param corpus_indices: 样本索引列表[0,....,29]
    :param batch_size:  小批量样本数2
    :param num_steps: 为每个样本所包含的时间步数6
    :param ctx:
    :return:
    """
    # 减1是因为输出的索引是相应输入的索引加1  根据每个样本的时间步计算总样本数
    num_examples = (len(corpus_indices) - 1) // num_steps
    # 计算每次采样总样本数num_examples采集完成需要进行采集的次数,这里4个样本，每次2个就需要采集2轮完成
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    # 随机打乱样本顺序
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield np.array(X, ctx), np.array(Y, ctx)

# 输入一个从0到29的连续整数的人工序列
my_seq = list(range(30))
# 批量大小和时间步数分别为2和6
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
"""
结论：相邻的两个随机小批量在原始序列上的位置不一定相毗邻
X:  [[12 13 14 15 16 17]
 [18 19 20 21 22 23]] 
Y: [[13 14 15 16 17 18]
 [19 20 21 22 23 24]] 

X:  [[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]] 
Y: [[ 1  2  3  4  5  6]
 [ 7  8  9 10 11 12]] 

"""

print("-------------------相邻采样-----------------------")
# 本函数已保存在d2lzh_tensorflow2包中方便以后使用
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = np.array(corpus_indices)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].reshape((
        batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


# 相邻的两个随机小批量在原始序列上的位置相毗邻
for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
