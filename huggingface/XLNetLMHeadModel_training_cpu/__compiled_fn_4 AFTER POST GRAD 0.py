from __future__ import annotations



def forward(self, primals_170: "f32[1024]", primals_176: "f32[1024]", primals_178: "f32[1024]", primals_184: "f32[1024]", primals_186: "f32[1024]", primals_192: "f32[1024]", primals_194: "f32[1024]", primals_200: "f32[1024]", primals_202: "f32[1024]", primals_208: "f32[1024]", primals_210: "f32[1024]", primals_216: "f32[1024]", primals_218: "f32[1024]", primals_224: "f32[1024]", primals_226: "f32[1024]", primals_232: "f32[1024]", primals_234: "f32[1024]", primals_240: "f32[1024]", primals_242: "f32[1024]", primals_248: "f32[1024]", primals_250: "f32[1024]", primals_256: "f32[1024]", primals_258: "f32[1024]", primals_264: "f32[1024]", primals_266: "f32[1024]", primals_272: "f32[1024]", primals_274: "f32[1024]", primals_280: "f32[1024]", primals_282: "f32[1024]", primals_288: "f32[1024]", primals_290: "f32[1024]", primals_296: "f32[1024]", primals_298: "f32[1024]", primals_304: "f32[1024]", primals_306: "f32[1024]", primals_312: "f32[1024]", primals_314: "f32[1024]", primals_320: "f32[1024]", primals_322: "f32[1024]", primals_328: "f32[1024]", primals_330: "f32[1024]", primals_336: "f32[1024]", primals_338: "f32[1024]", primals_344: "f32[1024]", primals_346: "f32[1024]", primals_352: "f32[1024]", primals_354: "f32[1024]", primals_360: "f32[1024]", primals_365: "i64[1, 512]", permute: "i64[512, 1]", getitem_1: "b8[512, 1, 1024]", iota: "i64[512]", getitem_5: "b8[1, 16, 512, 512]", getitem_7: "b8[512, 1, 1024]", mul_5: "f32[512, 1, 1024]", view_34: "f32[512, 1024]", addmm: "f32[512, 4096]", getitem_11: "b8[512, 1, 4096]", view_36: "f32[512, 4096]", getitem_13: "b8[512, 1, 1024]", mul_10: "f32[512, 1, 1024]", getitem_17: "b8[1, 16, 512, 512]", getitem_19: "b8[512, 1, 1024]", mul_13: "f32[512, 1, 1024]", view_72: "f32[512, 1024]", addmm_2: "f32[512, 4096]", getitem_23: "b8[512, 1, 4096]", view_74: "f32[512, 4096]", getitem_25: "b8[512, 1, 1024]", mul_18: "f32[512, 1, 1024]", getitem_29: "b8[1, 16, 512, 512]", getitem_31: "b8[512, 1, 1024]", mul_21: "f32[512, 1, 1024]", view_110: "f32[512, 1024]", addmm_4: "f32[512, 4096]", getitem_35: "b8[512, 1, 4096]", view_112: "f32[512, 4096]", getitem_37: "b8[512, 1, 1024]", mul_26: "f32[512, 1, 1024]", getitem_41: "b8[1, 16, 512, 512]", getitem_43: "b8[512, 1, 1024]", mul_29: "f32[512, 1, 1024]", view_148: "f32[512, 1024]", addmm_6: "f32[512, 4096]", getitem_47: "b8[512, 1, 4096]", view_150: "f32[512, 4096]", getitem_49: "b8[512, 1, 1024]", mul_34: "f32[512, 1, 1024]", getitem_53: "b8[1, 16, 512, 512]", getitem_55: "b8[512, 1, 1024]", mul_37: "f32[512, 1, 1024]", view_186: "f32[512, 1024]", addmm_8: "f32[512, 4096]", getitem_59: "b8[512, 1, 4096]", view_188: "f32[512, 4096]", getitem_61: "b8[512, 1, 1024]", mul_42: "f32[512, 1, 1024]", getitem_65: "b8[1, 16, 512, 512]", getitem_67: "b8[512, 1, 1024]", mul_45: "f32[512, 1, 1024]", view_224: "f32[512, 1024]", addmm_10: "f32[512, 4096]", getitem_71: "b8[512, 1, 4096]", view_226: "f32[512, 4096]", getitem_73: "b8[512, 1, 1024]", mul_50: "f32[512, 1, 1024]", getitem_77: "b8[1, 16, 512, 512]", getitem_79: "b8[512, 1, 1024]", mul_53: "f32[512, 1, 1024]", view_262: "f32[512, 1024]", addmm_12: "f32[512, 4096]", getitem_83: "b8[512, 1, 4096]", view_264: "f32[512, 4096]", getitem_85: "b8[512, 1, 1024]", mul_58: "f32[512, 1, 1024]", getitem_89: "b8[1, 16, 512, 512]", getitem_91: "b8[512, 1, 1024]", mul_61: "f32[512, 1, 1024]", view_300: "f32[512, 1024]", addmm_14: "f32[512, 4096]", getitem_95: "b8[512, 1, 4096]", view_302: "f32[512, 4096]", getitem_97: "b8[512, 1, 1024]", mul_66: "f32[512, 1, 1024]", getitem_101: "b8[1, 16, 512, 512]", getitem_103: "b8[512, 1, 1024]", mul_69: "f32[512, 1, 1024]", view_338: "f32[512, 1024]", addmm_16: "f32[512, 4096]", getitem_107: "b8[512, 1, 4096]", view_340: "f32[512, 4096]", getitem_109: "b8[512, 1, 1024]", mul_74: "f32[512, 1, 1024]", getitem_113: "b8[1, 16, 512, 512]", getitem_115: "b8[512, 1, 1024]", mul_77: "f32[512, 1, 1024]", view_376: "f32[512, 1024]", addmm_18: "f32[512, 4096]", getitem_119: "b8[512, 1, 4096]", view_378: "f32[512, 4096]", getitem_121: "b8[512, 1, 1024]", mul_82: "f32[512, 1, 1024]", getitem_125: "b8[1, 16, 512, 512]", getitem_127: "b8[512, 1, 1024]", mul_85: "f32[512, 1, 1024]", view_414: "f32[512, 1024]", addmm_20: "f32[512, 4096]", getitem_131: "b8[512, 1, 4096]", view_416: "f32[512, 4096]", getitem_133: "b8[512, 1, 1024]", mul_90: "f32[512, 1, 1024]", getitem_137: "b8[1, 16, 512, 512]", getitem_139: "b8[512, 1, 1024]", mul_93: "f32[512, 1, 1024]", view_452: "f32[512, 1024]", addmm_22: "f32[512, 4096]", getitem_143: "b8[512, 1, 4096]", view_454: "f32[512, 4096]", getitem_145: "b8[512, 1, 1024]", mul_98: "f32[512, 1, 1024]", getitem_149: "b8[1, 16, 512, 512]", getitem_151: "b8[512, 1, 1024]", mul_101: "f32[512, 1, 1024]", view_490: "f32[512, 1024]", addmm_24: "f32[512, 4096]", getitem_155: "b8[512, 1, 4096]", view_492: "f32[512, 4096]", getitem_157: "b8[512, 1, 1024]", mul_106: "f32[512, 1, 1024]", getitem_161: "b8[1, 16, 512, 512]", getitem_163: "b8[512, 1, 1024]", mul_109: "f32[512, 1, 1024]", view_528: "f32[512, 1024]", addmm_26: "f32[512, 4096]", getitem_167: "b8[512, 1, 4096]", view_530: "f32[512, 4096]", getitem_169: "b8[512, 1, 1024]", mul_114: "f32[512, 1, 1024]", getitem_173: "b8[1, 16, 512, 512]", getitem_175: "b8[512, 1, 1024]", mul_117: "f32[512, 1, 1024]", view_566: "f32[512, 1024]", addmm_28: "f32[512, 4096]", getitem_179: "b8[512, 1, 4096]", view_568: "f32[512, 4096]", getitem_181: "b8[512, 1, 1024]", mul_122: "f32[512, 1, 1024]", getitem_185: "b8[1, 16, 512, 512]", getitem_187: "b8[512, 1, 1024]", mul_125: "f32[512, 1, 1024]", view_604: "f32[512, 1024]", addmm_30: "f32[512, 4096]", getitem_191: "b8[512, 1, 4096]", view_606: "f32[512, 4096]", getitem_193: "b8[512, 1, 1024]", mul_130: "f32[512, 1, 1024]", getitem_197: "b8[1, 16, 512, 512]", getitem_199: "b8[512, 1, 1024]", mul_133: "f32[512, 1, 1024]", view_642: "f32[512, 1024]", addmm_32: "f32[512, 4096]", getitem_203: "b8[512, 1, 4096]", view_644: "f32[512, 4096]", getitem_205: "b8[512, 1, 1024]", mul_138: "f32[512, 1, 1024]", getitem_209: "b8[1, 16, 512, 512]", getitem_211: "b8[512, 1, 1024]", mul_141: "f32[512, 1, 1024]", view_680: "f32[512, 1024]", addmm_34: "f32[512, 4096]", getitem_215: "b8[512, 1, 4096]", view_682: "f32[512, 4096]", getitem_217: "b8[512, 1, 1024]", mul_146: "f32[512, 1, 1024]", getitem_221: "b8[1, 16, 512, 512]", getitem_223: "b8[512, 1, 1024]", mul_149: "f32[512, 1, 1024]", view_718: "f32[512, 1024]", addmm_36: "f32[512, 4096]", getitem_227: "b8[512, 1, 4096]", view_720: "f32[512, 4096]", getitem_229: "b8[512, 1, 1024]", mul_154: "f32[512, 1, 1024]", getitem_233: "b8[1, 16, 512, 512]", getitem_235: "b8[512, 1, 1024]", mul_157: "f32[512, 1, 1024]", view_756: "f32[512, 1024]", addmm_38: "f32[512, 4096]", getitem_239: "b8[512, 1, 4096]", view_758: "f32[512, 4096]", getitem_241: "b8[512, 1, 1024]", mul_162: "f32[512, 1, 1024]", getitem_245: "b8[1, 16, 512, 512]", getitem_247: "b8[512, 1, 1024]", mul_165: "f32[512, 1, 1024]", view_794: "f32[512, 1024]", addmm_40: "f32[512, 4096]", getitem_251: "b8[512, 1, 4096]", view_796: "f32[512, 4096]", getitem_253: "b8[512, 1, 1024]", mul_170: "f32[512, 1, 1024]", getitem_257: "b8[1, 16, 512, 512]", getitem_259: "b8[512, 1, 1024]", mul_173: "f32[512, 1, 1024]", view_832: "f32[512, 1024]", addmm_42: "f32[512, 4096]", getitem_263: "b8[512, 1, 4096]", view_834: "f32[512, 4096]", getitem_265: "b8[512, 1, 1024]", mul_178: "f32[512, 1, 1024]", getitem_269: "b8[1, 16, 512, 512]", getitem_271: "b8[512, 1, 1024]", mul_181: "f32[512, 1, 1024]", view_870: "f32[512, 1024]", addmm_44: "f32[512, 4096]", getitem_275: "b8[512, 1, 4096]", view_872: "f32[512, 4096]", getitem_277: "b8[512, 1, 1024]", mul_186: "f32[512, 1, 1024]", getitem_281: "b8[1, 16, 512, 512]", getitem_283: "b8[512, 1, 1024]", mul_189: "f32[512, 1, 1024]", view_908: "f32[512, 1024]", addmm_46: "f32[512, 4096]", getitem_287: "b8[512, 1, 4096]", view_910: "f32[512, 4096]", getitem_289: "b8[512, 1, 1024]", mul_194: "f32[512, 1, 1024]", getitem_293: "b8[512, 1, 1024]", view_912: "f32[512, 1024]", sub_73: "f32[512, 32000]", convert_element_type_4: "f32[]", permute_1013: "f32[32000, 1024]", div_27: "f32[512, 1, 1]", permute_1018: "f32[1024, 4096]", permute_1022: "f32[4096, 1024]", div_28: "f32[512, 1, 1]", permute_1027: "f32[1, 1024, 512]", permute_1028: "f32[1, 1024, 1024]", permute_1034: "f32[16, 512, 512]", permute_1035: "f32[16, 64, 512]", alias_26: "f32[1, 16, 512, 512]", permute_1041: "f32[16, 64, 512]", permute_1042: "f32[16, 1024, 64]", permute_1048: "f32[16, 64, 512]", permute_1049: "f32[16, 512, 64]", permute_1055: "f32[1, 1024, 1024]", permute_1059: "f32[1, 1024, 512]", permute_1060: "f32[1, 1024, 1024]", permute_1067: "f32[1, 1024, 1024]", permute_1074: "f32[1, 1024, 1024]", div_29: "f32[512, 1, 1]", permute_1079: "f32[1024, 4096]", permute_1083: "f32[4096, 1024]", div_30: "f32[512, 1, 1]", permute_1088: "f32[1, 1024, 512]", permute_1089: "f32[1, 1024, 1024]", permute_1095: "f32[16, 512, 512]", permute_1096: "f32[16, 64, 512]", alias_27: "f32[1, 16, 512, 512]", permute_1102: "f32[16, 64, 512]", permute_1103: "f32[16, 1024, 64]", permute_1109: "f32[16, 64, 512]", permute_1110: "f32[16, 512, 64]", permute_1120: "f32[1, 1024, 512]", permute_1121: "f32[1, 1024, 1024]", permute_1128: "f32[1, 1024, 1024]", permute_1135: "f32[1, 1024, 1024]", div_31: "f32[512, 1, 1]", permute_1140: "f32[1024, 4096]", permute_1144: "f32[4096, 1024]", div_32: "f32[512, 1, 1]", permute_1149: "f32[1, 1024, 512]", permute_1150: "f32[1, 1024, 1024]", permute_1156: "f32[16, 512, 512]", permute_1157: "f32[16, 64, 512]", alias_28: "f32[1, 16, 512, 512]", permute_1163: "f32[16, 64, 512]", permute_1164: "f32[16, 1024, 64]", permute_1170: "f32[16, 64, 512]", permute_1171: "f32[16, 512, 64]", permute_1181: "f32[1, 1024, 512]", permute_1182: "f32[1, 1024, 1024]", permute_1189: "f32[1, 1024, 1024]", permute_1196: "f32[1, 1024, 1024]", div_33: "f32[512, 1, 1]", permute_1201: "f32[1024, 4096]", permute_1205: "f32[4096, 1024]", div_34: "f32[512, 1, 1]", permute_1210: "f32[1, 1024, 512]", permute_1211: "f32[1, 1024, 1024]", permute_1217: "f32[16, 512, 512]", permute_1218: "f32[16, 64, 512]", alias_29: "f32[1, 16, 512, 512]", permute_1224: "f32[16, 64, 512]", permute_1225: "f32[16, 1024, 64]", permute_1231: "f32[16, 64, 512]", permute_1232: "f32[16, 512, 64]", permute_1242: "f32[1, 1024, 512]", permute_1243: "f32[1, 1024, 1024]", permute_1250: "f32[1, 1024, 1024]", permute_1257: "f32[1, 1024, 1024]", div_35: "f32[512, 1, 1]", permute_1262: "f32[1024, 4096]", permute_1266: "f32[4096, 1024]", div_36: "f32[512, 1, 1]", permute_1271: "f32[1, 1024, 512]", permute_1272: "f32[1, 1024, 1024]", permute_1278: "f32[16, 512, 512]", permute_1279: "f32[16, 64, 512]", alias_30: "f32[1, 16, 512, 512]", permute_1285: "f32[16, 64, 512]", permute_1286: "f32[16, 1024, 64]", permute_1292: "f32[16, 64, 512]", permute_1293: "f32[16, 512, 64]", permute_1303: "f32[1, 1024, 512]", permute_1304: "f32[1, 1024, 1024]", permute_1311: "f32[1, 1024, 1024]", permute_1318: "f32[1, 1024, 1024]", div_37: "f32[512, 1, 1]", permute_1323: "f32[1024, 4096]", permute_1327: "f32[4096, 1024]", div_38: "f32[512, 1, 1]", permute_1332: "f32[1, 1024, 512]", permute_1333: "f32[1, 1024, 1024]", permute_1339: "f32[16, 512, 512]", permute_1340: "f32[16, 64, 512]", alias_31: "f32[1, 16, 512, 512]", permute_1346: "f32[16, 64, 512]", permute_1347: "f32[16, 1024, 64]", permute_1353: "f32[16, 64, 512]", permute_1354: "f32[16, 512, 64]", permute_1364: "f32[1, 1024, 512]", permute_1365: "f32[1, 1024, 1024]", permute_1372: "f32[1, 1024, 1024]", permute_1379: "f32[1, 1024, 1024]", div_39: "f32[512, 1, 1]", permute_1384: "f32[1024, 4096]", permute_1388: "f32[4096, 1024]", div_40: "f32[512, 1, 1]", permute_1393: "f32[1, 1024, 512]", permute_1394: "f32[1, 1024, 1024]", permute_1400: "f32[16, 512, 512]", permute_1401: "f32[16, 64, 512]", alias_32: "f32[1, 16, 512, 512]", permute_1407: "f32[16, 64, 512]", permute_1408: "f32[16, 1024, 64]", permute_1414: "f32[16, 64, 512]", permute_1415: "f32[16, 512, 64]", permute_1425: "f32[1, 1024, 512]", permute_1426: "f32[1, 1024, 1024]", permute_1433: "f32[1, 1024, 1024]", permute_1440: "f32[1, 1024, 1024]", div_41: "f32[512, 1, 1]", permute_1445: "f32[1024, 4096]", permute_1449: "f32[4096, 1024]", div_42: "f32[512, 1, 1]", permute_1454: "f32[1, 1024, 512]", permute_1455: "f32[1, 1024, 1024]", permute_1461: "f32[16, 512, 512]", permute_1462: "f32[16, 64, 512]", alias_33: "f32[1, 16, 512, 512]", permute_1468: "f32[16, 64, 512]", permute_1469: "f32[16, 1024, 64]", permute_1475: "f32[16, 64, 512]", permute_1476: "f32[16, 512, 64]", permute_1486: "f32[1, 1024, 512]", permute_1487: "f32[1, 1024, 1024]", permute_1494: "f32[1, 1024, 1024]", permute_1501: "f32[1, 1024, 1024]", div_43: "f32[512, 1, 1]", permute_1506: "f32[1024, 4096]", permute_1510: "f32[4096, 1024]", div_44: "f32[512, 1, 1]", permute_1515: "f32[1, 1024, 512]", permute_1516: "f32[1, 1024, 1024]", permute_1522: "f32[16, 512, 512]", permute_1523: "f32[16, 64, 512]", alias_34: "f32[1, 16, 512, 512]", permute_1529: "f32[16, 64, 512]", permute_1530: "f32[16, 1024, 64]", permute_1536: "f32[16, 64, 512]", permute_1537: "f32[16, 512, 64]", permute_1547: "f32[1, 1024, 512]", permute_1548: "f32[1, 1024, 1024]", permute_1555: "f32[1, 1024, 1024]", permute_1562: "f32[1, 1024, 1024]", div_45: "f32[512, 1, 1]", permute_1567: "f32[1024, 4096]", permute_1571: "f32[4096, 1024]", div_46: "f32[512, 1, 1]", permute_1576: "f32[1, 1024, 512]", permute_1577: "f32[1, 1024, 1024]", permute_1583: "f32[16, 512, 512]", permute_1584: "f32[16, 64, 512]", alias_35: "f32[1, 16, 512, 512]", permute_1590: "f32[16, 64, 512]", permute_1591: "f32[16, 1024, 64]", permute_1597: "f32[16, 64, 512]", permute_1598: "f32[16, 512, 64]", permute_1608: "f32[1, 1024, 512]", permute_1609: "f32[1, 1024, 1024]", permute_1616: "f32[1, 1024, 1024]", permute_1623: "f32[1, 1024, 1024]", div_47: "f32[512, 1, 1]", permute_1628: "f32[1024, 4096]", permute_1632: "f32[4096, 1024]", div_48: "f32[512, 1, 1]", permute_1637: "f32[1, 1024, 512]", permute_1638: "f32[1, 1024, 1024]", permute_1644: "f32[16, 512, 512]", permute_1645: "f32[16, 64, 512]", alias_36: "f32[1, 16, 512, 512]", permute_1651: "f32[16, 64, 512]", permute_1652: "f32[16, 1024, 64]", permute_1658: "f32[16, 64, 512]", permute_1659: "f32[16, 512, 64]", permute_1669: "f32[1, 1024, 512]", permute_1670: "f32[1, 1024, 1024]", permute_1677: "f32[1, 1024, 1024]", permute_1684: "f32[1, 1024, 1024]", div_49: "f32[512, 1, 1]", permute_1689: "f32[1024, 4096]", permute_1693: "f32[4096, 1024]", div_50: "f32[512, 1, 1]", permute_1698: "f32[1, 1024, 512]", permute_1699: "f32[1, 1024, 1024]", permute_1705: "f32[16, 512, 512]", permute_1706: "f32[16, 64, 512]", alias_37: "f32[1, 16, 512, 512]", permute_1712: "f32[16, 64, 512]", permute_1713: "f32[16, 1024, 64]", permute_1719: "f32[16, 64, 512]", permute_1720: "f32[16, 512, 64]", permute_1730: "f32[1, 1024, 512]", permute_1731: "f32[1, 1024, 1024]", permute_1738: "f32[1, 1024, 1024]", permute_1745: "f32[1, 1024, 1024]", div_51: "f32[512, 1, 1]", permute_1750: "f32[1024, 4096]", permute_1754: "f32[4096, 1024]", div_52: "f32[512, 1, 1]", permute_1759: "f32[1, 1024, 512]", permute_1760: "f32[1, 1024, 1024]", permute_1766: "f32[16, 512, 512]", permute_1767: "f32[16, 64, 512]", alias_38: "f32[1, 16, 512, 512]", permute_1773: "f32[16, 64, 512]", permute_1774: "f32[16, 1024, 64]", permute_1780: "f32[16, 64, 512]", permute_1781: "f32[16, 512, 64]", permute_1791: "f32[1, 1024, 512]", permute_1792: "f32[1, 1024, 1024]", permute_1799: "f32[1, 1024, 1024]", permute_1806: "f32[1, 1024, 1024]", div_53: "f32[512, 1, 1]", permute_1811: "f32[1024, 4096]", permute_1815: "f32[4096, 1024]", div_54: "f32[512, 1, 1]", permute_1820: "f32[1, 1024, 512]", permute_1821: "f32[1, 1024, 1024]", permute_1827: "f32[16, 512, 512]", permute_1828: "f32[16, 64, 512]", alias_39: "f32[1, 16, 512, 512]", permute_1834: "f32[16, 64, 512]", permute_1835: "f32[16, 1024, 64]", permute_1841: "f32[16, 64, 512]", permute_1842: "f32[16, 512, 64]", permute_1852: "f32[1, 1024, 512]", permute_1853: "f32[1, 1024, 1024]", permute_1860: "f32[1, 1024, 1024]", permute_1867: "f32[1, 1024, 1024]", div_55: "f32[512, 1, 1]", permute_1872: "f32[1024, 4096]", permute_1876: "f32[4096, 1024]", div_56: "f32[512, 1, 1]", permute_1881: "f32[1, 1024, 512]", permute_1882: "f32[1, 1024, 1024]", permute_1888: "f32[16, 512, 512]", permute_1889: "f32[16, 64, 512]", alias_40: "f32[1, 16, 512, 512]", permute_1895: "f32[16, 64, 512]", permute_1896: "f32[16, 1024, 64]", permute_1902: "f32[16, 64, 512]", permute_1903: "f32[16, 512, 64]", permute_1913: "f32[1, 1024, 512]", permute_1914: "f32[1, 1024, 1024]", permute_1921: "f32[1, 1024, 1024]", permute_1928: "f32[1, 1024, 1024]", div_57: "f32[512, 1, 1]", permute_1933: "f32[1024, 4096]", permute_1937: "f32[4096, 1024]", div_58: "f32[512, 1, 1]", permute_1942: "f32[1, 1024, 512]", permute_1943: "f32[1, 1024, 1024]", permute_1949: "f32[16, 512, 512]", permute_1950: "f32[16, 64, 512]", alias_41: "f32[1, 16, 512, 512]", permute_1956: "f32[16, 64, 512]", permute_1957: "f32[16, 1024, 64]", permute_1963: "f32[16, 64, 512]", permute_1964: "f32[16, 512, 64]", permute_1974: "f32[1, 1024, 512]", permute_1975: "f32[1, 1024, 1024]", permute_1982: "f32[1, 1024, 1024]", permute_1989: "f32[1, 1024, 1024]", div_59: "f32[512, 1, 1]", permute_1994: "f32[1024, 4096]", permute_1998: "f32[4096, 1024]", div_60: "f32[512, 1, 1]", permute_2003: "f32[1, 1024, 512]", permute_2004: "f32[1, 1024, 1024]", permute_2010: "f32[16, 512, 512]", permute_2011: "f32[16, 64, 512]", alias_42: "f32[1, 16, 512, 512]", permute_2017: "f32[16, 64, 512]", permute_2018: "f32[16, 1024, 64]", permute_2024: "f32[16, 64, 512]", permute_2025: "f32[16, 512, 64]", permute_2035: "f32[1, 1024, 512]", permute_2036: "f32[1, 1024, 1024]", permute_2043: "f32[1, 1024, 1024]", permute_2050: "f32[1, 1024, 1024]", div_61: "f32[512, 1, 1]", permute_2055: "f32[1024, 4096]", permute_2059: "f32[4096, 1024]", div_62: "f32[512, 1, 1]", permute_2064: "f32[1, 1024, 512]", permute_2065: "f32[1, 1024, 1024]", permute_2071: "f32[16, 512, 512]", permute_2072: "f32[16, 64, 512]", alias_43: "f32[1, 16, 512, 512]", permute_2078: "f32[16, 64, 512]", permute_2079: "f32[16, 1024, 64]", permute_2085: "f32[16, 64, 512]", permute_2086: "f32[16, 512, 64]", permute_2096: "f32[1, 1024, 512]", permute_2097: "f32[1, 1024, 1024]", permute_2104: "f32[1, 1024, 1024]", permute_2111: "f32[1, 1024, 1024]", div_63: "f32[512, 1, 1]", permute_2116: "f32[1024, 4096]", permute_2120: "f32[4096, 1024]", div_64: "f32[512, 1, 1]", permute_2125: "f32[1, 1024, 512]", permute_2126: "f32[1, 1024, 1024]", permute_2132: "f32[16, 512, 512]", permute_2133: "f32[16, 64, 512]", alias_44: "f32[1, 16, 512, 512]", permute_2139: "f32[16, 64, 512]", permute_2140: "f32[16, 1024, 64]", permute_2146: "f32[16, 64, 512]", permute_2147: "f32[16, 512, 64]", permute_2157: "f32[1, 1024, 512]", permute_2158: "f32[1, 1024, 1024]", permute_2165: "f32[1, 1024, 1024]", permute_2172: "f32[1, 1024, 1024]", div_65: "f32[512, 1, 1]", permute_2177: "f32[1024, 4096]", permute_2181: "f32[4096, 1024]", div_66: "f32[512, 1, 1]", permute_2186: "f32[1, 1024, 512]", permute_2187: "f32[1, 1024, 1024]", permute_2193: "f32[16, 512, 512]", permute_2194: "f32[16, 64, 512]", alias_45: "f32[1, 16, 512, 512]", permute_2200: "f32[16, 64, 512]", permute_2201: "f32[16, 1024, 64]", permute_2207: "f32[16, 64, 512]", permute_2208: "f32[16, 512, 64]", permute_2218: "f32[1, 1024, 512]", permute_2219: "f32[1, 1024, 1024]", permute_2226: "f32[1, 1024, 1024]", permute_2233: "f32[1, 1024, 1024]", div_67: "f32[512, 1, 1]", permute_2238: "f32[1024, 4096]", permute_2242: "f32[4096, 1024]", div_68: "f32[512, 1, 1]", permute_2247: "f32[1, 1024, 512]", permute_2248: "f32[1, 1024, 1024]", permute_2254: "f32[16, 512, 512]", permute_2255: "f32[16, 64, 512]", alias_46: "f32[1, 16, 512, 512]", permute_2261: "f32[16, 64, 512]", permute_2262: "f32[16, 1024, 64]", permute_2268: "f32[16, 64, 512]", permute_2269: "f32[16, 512, 64]", permute_2279: "f32[1, 1024, 512]", permute_2280: "f32[1, 1024, 1024]", permute_2287: "f32[1, 1024, 1024]", permute_2294: "f32[1, 1024, 1024]", div_69: "f32[512, 1, 1]", permute_2299: "f32[1024, 4096]", permute_2303: "f32[4096, 1024]", div_70: "f32[512, 1, 1]", permute_2308: "f32[1, 1024, 512]", permute_2309: "f32[1, 1024, 1024]", permute_2315: "f32[16, 512, 512]", permute_2316: "f32[16, 64, 512]", alias_47: "f32[1, 16, 512, 512]", permute_2322: "f32[16, 64, 512]", permute_2323: "f32[16, 1024, 64]", permute_2329: "f32[16, 64, 512]", permute_2330: "f32[16, 512, 64]", permute_2340: "f32[1, 1024, 512]", permute_2341: "f32[1, 1024, 1024]", permute_2348: "f32[1, 1024, 1024]", permute_2355: "f32[1, 1024, 1024]", div_71: "f32[512, 1, 1]", permute_2360: "f32[1024, 4096]", permute_2364: "f32[4096, 1024]", div_72: "f32[512, 1, 1]", permute_2369: "f32[1, 1024, 512]", permute_2370: "f32[1, 1024, 1024]", permute_2376: "f32[16, 512, 512]", permute_2377: "f32[16, 64, 512]", alias_48: "f32[1, 16, 512, 512]", permute_2383: "f32[16, 64, 512]", permute_2384: "f32[16, 1024, 64]", permute_2390: "f32[16, 64, 512]", permute_2391: "f32[16, 512, 64]", permute_2401: "f32[1, 1024, 512]", permute_2402: "f32[1, 1024, 1024]", permute_2409: "f32[1, 1024, 1024]", permute_2416: "f32[1, 1024, 1024]", div_73: "f32[512, 1, 1]", permute_2421: "f32[1024, 4096]", permute_2425: "f32[4096, 1024]", div_74: "f32[512, 1, 1]", permute_2430: "f32[1, 1024, 512]", permute_2431: "f32[1, 1024, 1024]", permute_2437: "f32[16, 512, 512]", permute_2438: "f32[16, 64, 512]", alias_49: "f32[1, 16, 512, 512]", permute_2444: "f32[16, 64, 512]", permute_2445: "f32[16, 1024, 64]", permute_2451: "f32[16, 64, 512]", permute_2452: "f32[16, 512, 64]", permute_2462: "f32[1, 1024, 512]", permute_2463: "f32[1, 1024, 1024]", permute_2470: "f32[1, 1024, 1024]", permute_2477: "f32[1, 1024, 1024]", tangents_1: "f32[]", tangents_2: "f32[1, 512, 32000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_35: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm, [512, 1, 4096]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_8: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_35, 0.7071067811865476)
    erf: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_9: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_73: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_2, [512, 1, 4096]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_16: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_73, 0.7071067811865476)
    erf_1: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_16);  mul_16 = None
    add_20: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_111: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_4, [512, 1, 4096]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_24: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_111, 0.7071067811865476)
    erf_2: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
    add_31: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_149: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_6, [512, 1, 4096]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_32: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_149, 0.7071067811865476)
    erf_3: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_32);  mul_32 = None
    add_42: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_187: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_8, [512, 1, 4096]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_40: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_187, 0.7071067811865476)
    erf_4: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_53: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_225: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_10, [512, 1, 4096]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_48: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_225, 0.7071067811865476)
    erf_5: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_64: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_263: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_12, [512, 1, 4096]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_56: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476)
    erf_6: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
    add_75: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_301: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_14, [512, 1, 4096]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_64: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_301, 0.7071067811865476)
    erf_7: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
    add_86: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_339: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_16, [512, 1, 4096]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_72: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_339, 0.7071067811865476)
    erf_8: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
    add_97: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_377: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_18, [512, 1, 4096]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_80: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_377, 0.7071067811865476)
    erf_9: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_80);  mul_80 = None
    add_108: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_415: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_20, [512, 1, 4096]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_88: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476)
    erf_10: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_119: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_453: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_22, [512, 1, 4096]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_96: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_453, 0.7071067811865476)
    erf_11: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
    add_130: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_491: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_24, [512, 1, 4096]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_104: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_491, 0.7071067811865476)
    erf_12: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_104);  mul_104 = None
    add_141: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_529: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_26, [512, 1, 4096]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_112: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_529, 0.7071067811865476)
    erf_13: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
    add_152: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_567: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_28, [512, 1, 4096]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_120: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_567, 0.7071067811865476)
    erf_14: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_120);  mul_120 = None
    add_163: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_605: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_30, [512, 1, 4096]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_128: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_605, 0.7071067811865476)
    erf_15: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_128);  mul_128 = None
    add_174: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_643: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_32, [512, 1, 4096]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_136: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_643, 0.7071067811865476)
    erf_16: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_136);  mul_136 = None
    add_185: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_681: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_34, [512, 1, 4096]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_144: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_681, 0.7071067811865476)
    erf_17: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_144);  mul_144 = None
    add_196: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_719: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_36, [512, 1, 4096]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_152: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_719, 0.7071067811865476)
    erf_18: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_152);  mul_152 = None
    add_207: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_757: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_38, [512, 1, 4096]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_160: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_757, 0.7071067811865476)
    erf_19: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_160);  mul_160 = None
    add_218: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_795: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_40, [512, 1, 4096]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_168: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_795, 0.7071067811865476)
    erf_20: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
    add_229: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_833: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_42, [512, 1, 4096]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_176: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_833, 0.7071067811865476)
    erf_21: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_176);  mul_176 = None
    add_240: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_871: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_44, [512, 1, 4096]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_184: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_871, 0.7071067811865476)
    erf_22: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_184);  mul_184 = None
    add_251: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_909: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(addmm_46, [512, 1, 4096]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_192: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_909, 0.7071067811865476)
    erf_23: "f32[512, 1, 4096]" = torch.ops.aten.erf.default(mul_192);  mul_192 = None
    add_262: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1469, code: loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    view_915: "i64[512]" = torch.ops.aten.reshape.default(primals_365, [-1]);  primals_365 = None
    full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    div_26: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_4);  tangents_1 = convert_element_type_4 = None
    unsqueeze_604: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(view_915, 1);  view_915 = None
    ne_3: "b8[512, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_604, -100)
    where_2: "i64[512, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_604, full_default);  unsqueeze_604 = full_default = None
    full_default_3: "f32[512, 32000]" = torch.ops.aten.full.default([512, 32000], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[512, 32000]" = torch.ops.aten.scatter.value(full_default_3, 1, where_2, -1.0);  full_default_3 = where_2 = None
    where_3: "f32[512, 1]" = torch.ops.aten.where.self(ne_3, div_26, full_default_1);  ne_3 = div_26 = None
    mul_196: "f32[512, 32000]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    exp_25: "f32[512, 32000]" = torch.ops.aten.exp.default(sub_73);  sub_73 = None
    sum_28: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(mul_196, [1], True)
    mul_197: "f32[512, 32000]" = torch.ops.aten.mul.Tensor(exp_25, sum_28);  exp_25 = sum_28 = None
    sub_74: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
    view_916: "f32[1, 512, 32000]" = torch.ops.aten.reshape.default(sub_74, [1, 512, 32000]);  sub_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1469, code: loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    add_266: "f32[1, 512, 32000]" = torch.ops.aten.add.Tensor(tangents_2, view_916);  tangents_2 = view_916 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1463, code: logits = self.lm_loss(transformer_outputs[0])
    view_917: "f32[512, 32000]" = torch.ops.aten.reshape.default(add_266, [512, 32000]);  add_266 = None
    mm: "f32[512, 1024]" = torch.ops.aten.mm.default(view_917, permute_1013);  permute_1013 = None
    permute_1014: "f32[32000, 512]" = torch.ops.aten.permute.default(view_917, [1, 0])
    mm_1: "f32[32000, 1024]" = torch.ops.aten.mm.default(permute_1014, view_912);  permute_1014 = view_912 = None
    permute_1015: "f32[1024, 32000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_29: "f32[1, 32000]" = torch.ops.aten.sum.dim_IntList(view_917, [0], True);  view_917 = None
    view_918: "f32[32000]" = torch.ops.aten.reshape.default(sum_29, [32000]);  sum_29 = None
    permute_1016: "f32[32000, 1024]" = torch.ops.aten.permute.default(permute_1015, [1, 0]);  permute_1015 = None
    view_919: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm, [1, 512, 1024]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1260, code: output = output.permute(1, 0, 2).contiguous()
    permute_1017: "f32[512, 1, 1024]" = torch.ops.aten.permute.default(view_919, [1, 0, 2]);  view_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1257, code: output = self.dropout(output_g if output_g is not None else output_h)
    convert_element_type_5: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_293, torch.float32);  getitem_293 = None
    mul_198: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_199: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(permute_1017, mul_198);  permute_1017 = mul_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_201: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_199, primals_360);  primals_360 = None
    mul_202: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_201, 1024)
    sum_30: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_201, [2], True)
    mul_203: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_201, mul_194);  mul_201 = None
    sum_31: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [2], True);  mul_203 = None
    mul_204: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_194, sum_31);  sum_31 = None
    sub_76: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_202, sum_30);  mul_202 = sum_30 = None
    sub_77: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_76, mul_204);  sub_76 = mul_204 = None
    mul_205: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_27, sub_77);  div_27 = sub_77 = None
    mul_206: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_199, mul_194);  mul_194 = None
    sum_32: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_206, [0, 1]);  mul_206 = None
    sum_33: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_199, [0, 1]);  mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_6: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_289, torch.float32);  getitem_289 = None
    mul_207: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
    mul_208: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_205, mul_207);  mul_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_920: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_208, [512, 1024]);  mul_208 = None
    mm_2: "f32[512, 4096]" = torch.ops.aten.mm.default(view_920, permute_1018);  permute_1018 = None
    permute_1019: "f32[1024, 512]" = torch.ops.aten.permute.default(view_920, [1, 0])
    mm_3: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1019, view_910);  permute_1019 = view_910 = None
    permute_1020: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_34: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_920, [0], True);  view_920 = None
    view_921: "f32[1024]" = torch.ops.aten.reshape.default(sum_34, [1024]);  sum_34 = None
    permute_1021: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1020, [1, 0]);  permute_1020 = None
    view_922: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_2, [512, 1, 4096]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_7: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_287, torch.float32);  getitem_287 = None
    mul_209: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
    mul_210: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_922, mul_209);  view_922 = mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_212: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_262, 0.5);  add_262 = None
    mul_213: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_909, view_909)
    mul_214: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_213, -0.5);  mul_213 = None
    exp_26: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_214);  mul_214 = None
    mul_215: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_216: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_909, mul_215);  view_909 = mul_215 = None
    add_268: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_212, mul_216);  mul_212 = mul_216 = None
    mul_217: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_210, add_268);  mul_210 = add_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_923: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_217, [512, 4096]);  mul_217 = None
    mm_4: "f32[512, 1024]" = torch.ops.aten.mm.default(view_923, permute_1022);  permute_1022 = None
    permute_1023: "f32[4096, 512]" = torch.ops.aten.permute.default(view_923, [1, 0])
    mm_5: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1023, view_908);  permute_1023 = view_908 = None
    permute_1024: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_35: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_923, [0], True);  view_923 = None
    view_924: "f32[4096]" = torch.ops.aten.reshape.default(sum_35, [4096]);  sum_35 = None
    permute_1025: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1024, [1, 0]);  permute_1024 = None
    view_925: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_4, [512, 1, 1024]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_269: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_205, view_925);  mul_205 = view_925 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_219: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_269, primals_354);  primals_354 = None
    mul_220: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_219, 1024)
    sum_36: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_219, [2], True)
    mul_221: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_219, mul_189);  mul_219 = None
    sum_37: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_221, [2], True);  mul_221 = None
    mul_222: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_189, sum_37);  sum_37 = None
    sub_79: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_220, sum_36);  mul_220 = sum_36 = None
    sub_80: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_79, mul_222);  sub_79 = mul_222 = None
    mul_223: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_28, sub_80);  div_28 = sub_80 = None
    mul_224: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_269, mul_189);  mul_189 = None
    sum_38: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_224, [0, 1]);  mul_224 = None
    sum_39: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_269, [0, 1]);  add_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_8: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_283, torch.float32);  getitem_283 = None
    mul_225: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_226: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_223, mul_225);  mul_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_926: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_226, [512, 1, 1024, 1, 1]);  mul_226 = None
    permute_1026: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_926, [0, 3, 4, 1, 2]);  view_926 = None
    view_927: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1026, [1, 512, 1024]);  permute_1026 = None
    bmm_192: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1027, view_927);  permute_1027 = None
    bmm_193: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_927, permute_1028);  view_927 = permute_1028 = None
    view_928: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_192, [64, 16, 1, 1024, 1]);  bmm_192 = None
    permute_1029: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_928, [4, 2, 3, 0, 1]);  view_928 = None
    view_929: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_193, [512, 64, 16, 1, 1]);  bmm_193 = None
    permute_1030: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_929, [0, 3, 4, 1, 2]);  view_929 = None
    permute_1031: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1029, [2, 4, 3, 0, 1]);  permute_1029 = None
    squeeze_1: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1031, 4);  permute_1031 = None
    squeeze_2: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_1, 3);  squeeze_1 = None
    permute_1032: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1030, [0, 1, 4, 3, 2]);  permute_1030 = None
    squeeze_3: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1032, 4);  permute_1032 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_930: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_3, [512, 1, 16, 64, 1]);  squeeze_3 = None
    permute_1033: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_930, [2, 0, 4, 1, 3]);  view_930 = None
    view_931: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_1033, [16, 512, 64]);  permute_1033 = None
    bmm_194: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_1034, view_931);  permute_1034 = None
    bmm_195: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_931, permute_1035);  view_931 = permute_1035 = None
    view_932: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_194, [16, 512, 1, 64, 1]);  bmm_194 = None
    permute_1036: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_932, [4, 2, 0, 3, 1]);  view_932 = None
    view_933: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_195, [16, 512, 512, 1, 1]);  bmm_195 = None
    permute_1037: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_933, [1, 3, 0, 4, 2]);  view_933 = None
    permute_1038: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1036, [4, 1, 2, 3, 0]);  permute_1036 = None
    squeeze_4: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1038, 4);  permute_1038 = None
    permute_1039: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_1037, [1, 2, 0, 4, 3]);  permute_1037 = None
    squeeze_5: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_1039, 4);  permute_1039 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_9: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_281, torch.float32);  getitem_281 = None
    mul_227: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
    mul_228: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_5, mul_227);  squeeze_5 = mul_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_229: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_228, alias_26);  mul_228 = None
    sum_40: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [3], True)
    mul_230: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_26, sum_40);  alias_26 = sum_40 = None
    sub_81: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_229, mul_230);  mul_229 = mul_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_231: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_81, 0.125);  sub_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    full_default_5: "f32[1, 16, 512, 1023]" = torch.ops.aten.full.default([1, 16, 512, 1023], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_231, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_934: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put, [1, 16, 1023, 512]);  index_put = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    full_default_6: "f32[1, 16, 1023, 512]" = torch.ops.aten.full.default([1, 16, 1023, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    full_default_7: "f32[1, 16, 1024, 512]" = torch.ops.aten.full.default([1, 16, 1024, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_1: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_934, 2, 1, 9223372036854775807);  view_934 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_935: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_1, [1, 16, 512, 1024]);  slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_936: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_935, [1, 16, 512, 1024, 1]);  view_935 = None
    permute_1040: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_936, [1, 2, 4, 0, 3]);  view_936 = None
    view_937: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_1040, [16, 512, 1024]);  permute_1040 = None
    bmm_196: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_1041, view_937);  permute_1041 = None
    bmm_197: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_937, permute_1042);  view_937 = permute_1042 = None
    view_938: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_196, [16, 64, 1, 1024, 1]);  bmm_196 = None
    permute_1043: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_938, [2, 0, 4, 3, 1]);  view_938 = None
    view_939: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_197, [16, 512, 64, 1, 1]);  bmm_197 = None
    permute_1044: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_939, [3, 0, 1, 4, 2]);  view_939 = None
    permute_1045: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1043, [3, 0, 1, 4, 2]);  permute_1043 = None
    squeeze_6: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1045, 4);  permute_1045 = None
    permute_1046: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1044, [2, 0, 1, 4, 3]);  permute_1044 = None
    squeeze_7: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1046, 4);  permute_1046 = None
    sum_41: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_7, [0, 1], True)
    view_940: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_41, [16, 64]);  sum_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_941: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_231, [1, 16, 512, 512, 1]);  mul_231 = None
    permute_1047: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_941, [1, 2, 4, 0, 3]);  view_941 = None
    view_942: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_1047, [16, 512, 512]);  permute_1047 = None
    bmm_198: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_1048, view_942);  permute_1048 = None
    bmm_199: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_942, permute_1049);  view_942 = permute_1049 = None
    view_943: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_198, [16, 64, 1, 512, 1]);  bmm_198 = None
    permute_1050: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_943, [2, 0, 4, 3, 1]);  view_943 = None
    view_944: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_199, [16, 512, 64, 1, 1]);  bmm_199 = None
    permute_1051: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_944, [3, 0, 1, 4, 2]);  view_944 = None
    permute_1052: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1050, [3, 0, 1, 4, 2]);  permute_1050 = None
    squeeze_8: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1052, 4);  permute_1052 = None
    permute_1053: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1051, [2, 0, 1, 4, 3]);  permute_1051 = None
    squeeze_9: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1053, 4);  permute_1053 = None
    sum_42: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_9, [0, 1], True)
    view_945: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_42, [16, 64]);  sum_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_270: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_7, squeeze_9);  squeeze_7 = squeeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_946: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_6, [1024, 1, 16, 64, 1]);  squeeze_6 = None
    permute_1054: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_946, [0, 4, 1, 2, 3]);  view_946 = None
    view_947: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_1054, [1, 1024, 1024]);  permute_1054 = None
    bmm_200: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_947);  view_947 = None
    view_948: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_200, [1024, 1, 16, 64, 1]);  bmm_200 = None
    permute_1056: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_948, [4, 1, 2, 3, 0]);  view_948 = None
    permute_1057: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1056, [4, 2, 3, 0, 1]);  permute_1056 = None
    squeeze_10: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1057, 4);  permute_1057 = None
    squeeze_11: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_10, 3);  squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_949: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_4, [512, 1, 16, 64, 1]);  squeeze_4 = None
    permute_1058: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_949, [0, 4, 1, 2, 3]);  view_949 = None
    clone_53: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1058, memory_format = torch.contiguous_format);  permute_1058 = None
    view_950: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_53, [1, 512, 1024]);  clone_53 = None
    bmm_201: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1059, view_950)
    bmm_202: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_950, permute_1060);  view_950 = permute_1060 = None
    view_951: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_201, [1024, 1, 16, 64, 1]);  bmm_201 = None
    permute_1061: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_951, [4, 1, 2, 3, 0]);  view_951 = None
    view_952: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_202, [512, 1024, 1, 1, 1]);  bmm_202 = None
    permute_1062: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_952, [0, 2, 3, 4, 1]);  view_952 = None
    permute_1063: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1061, [4, 2, 3, 0, 1]);  permute_1061 = None
    squeeze_12: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1063, 4);  permute_1063 = None
    squeeze_13: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_12, 3);  squeeze_12 = None
    permute_1064: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1062, [0, 1, 4, 2, 3]);  permute_1062 = None
    squeeze_14: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1064, 4);  permute_1064 = None
    squeeze_15: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_14, 3);  squeeze_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_271: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_223, squeeze_15);  mul_223 = squeeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_953: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_8, [512, 1, 16, 64, 1]);  squeeze_8 = None
    permute_1065: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_953, [0, 4, 1, 2, 3]);  view_953 = None
    view_954: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1065, [1, 512, 1024]);  permute_1065 = None
    bmm_203: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1059, view_954)
    bmm_204: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_954, permute_1067);  view_954 = permute_1067 = None
    view_955: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_203, [1024, 1, 16, 64, 1]);  bmm_203 = None
    permute_1068: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_955, [4, 1, 2, 3, 0]);  view_955 = None
    view_956: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_204, [512, 1024, 1, 1, 1]);  bmm_204 = None
    permute_1069: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_956, [0, 2, 3, 4, 1]);  view_956 = None
    permute_1070: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1068, [4, 2, 3, 0, 1]);  permute_1068 = None
    squeeze_16: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1070, 4);  permute_1070 = None
    squeeze_17: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_16, 3);  squeeze_16 = None
    permute_1071: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1069, [0, 1, 4, 2, 3]);  permute_1069 = None
    squeeze_18: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1071, 4);  permute_1071 = None
    squeeze_19: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_18, 3);  squeeze_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_272: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_271, squeeze_19);  add_271 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_957: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_270, [512, 1, 16, 64, 1]);  add_270 = None
    permute_1072: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_957, [0, 4, 1, 2, 3]);  view_957 = None
    clone_54: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1072, memory_format = torch.contiguous_format);  permute_1072 = None
    view_958: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_54, [1, 512, 1024]);  clone_54 = None
    bmm_205: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1059, view_958);  permute_1059 = None
    bmm_206: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_958, permute_1074);  view_958 = permute_1074 = None
    view_959: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_205, [1024, 1, 16, 64, 1]);  bmm_205 = None
    permute_1075: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_959, [4, 1, 2, 3, 0]);  view_959 = None
    view_960: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_206, [512, 1024, 1, 1, 1]);  bmm_206 = None
    permute_1076: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_960, [0, 2, 3, 4, 1]);  view_960 = None
    permute_1077: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1075, [4, 2, 3, 0, 1]);  permute_1075 = None
    squeeze_20: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1077, 4);  permute_1077 = None
    squeeze_21: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_20, 3);  squeeze_20 = None
    permute_1078: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1076, [0, 1, 4, 2, 3]);  permute_1076 = None
    squeeze_22: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1078, 4);  permute_1078 = None
    squeeze_23: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_22, 3);  squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_273: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_272, squeeze_23);  add_272 = squeeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_233: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_273, primals_352);  primals_352 = None
    mul_234: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_233, 1024)
    sum_43: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_233, [2], True)
    mul_235: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_233, mul_186);  mul_233 = None
    sum_44: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_235, [2], True);  mul_235 = None
    mul_236: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_186, sum_44);  sum_44 = None
    sub_83: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_234, sum_43);  mul_234 = sum_43 = None
    sub_84: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_83, mul_236);  sub_83 = mul_236 = None
    mul_237: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_29, sub_84);  div_29 = sub_84 = None
    mul_238: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_273, mul_186);  mul_186 = None
    sum_45: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_238, [0, 1]);  mul_238 = None
    sum_46: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_273, [0, 1]);  add_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_10: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_277, torch.float32);  getitem_277 = None
    mul_239: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_240: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_237, mul_239);  mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_961: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_240, [512, 1024]);  mul_240 = None
    mm_6: "f32[512, 4096]" = torch.ops.aten.mm.default(view_961, permute_1079);  permute_1079 = None
    permute_1080: "f32[1024, 512]" = torch.ops.aten.permute.default(view_961, [1, 0])
    mm_7: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1080, view_872);  permute_1080 = view_872 = None
    permute_1081: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_47: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_961, [0], True);  view_961 = None
    view_962: "f32[1024]" = torch.ops.aten.reshape.default(sum_47, [1024]);  sum_47 = None
    permute_1082: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1081, [1, 0]);  permute_1081 = None
    view_963: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_6, [512, 1, 4096]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_11: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_275, torch.float32);  getitem_275 = None
    mul_241: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_242: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_963, mul_241);  view_963 = mul_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_244: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_251, 0.5);  add_251 = None
    mul_245: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_871, view_871)
    mul_246: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_245, -0.5);  mul_245 = None
    exp_27: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_246);  mul_246 = None
    mul_247: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_248: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_871, mul_247);  view_871 = mul_247 = None
    add_275: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_244, mul_248);  mul_244 = mul_248 = None
    mul_249: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_242, add_275);  mul_242 = add_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_964: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_249, [512, 4096]);  mul_249 = None
    mm_8: "f32[512, 1024]" = torch.ops.aten.mm.default(view_964, permute_1083);  permute_1083 = None
    permute_1084: "f32[4096, 512]" = torch.ops.aten.permute.default(view_964, [1, 0])
    mm_9: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1084, view_870);  permute_1084 = view_870 = None
    permute_1085: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_48: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_964, [0], True);  view_964 = None
    view_965: "f32[4096]" = torch.ops.aten.reshape.default(sum_48, [4096]);  sum_48 = None
    permute_1086: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1085, [1, 0]);  permute_1085 = None
    view_966: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_8, [512, 1, 1024]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_276: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_237, view_966);  mul_237 = view_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_251: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_276, primals_346);  primals_346 = None
    mul_252: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_251, 1024)
    sum_49: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True)
    mul_253: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_251, mul_181);  mul_251 = None
    sum_50: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True);  mul_253 = None
    mul_254: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_181, sum_50);  sum_50 = None
    sub_86: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_252, sum_49);  mul_252 = sum_49 = None
    sub_87: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_86, mul_254);  sub_86 = mul_254 = None
    mul_255: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_30, sub_87);  div_30 = sub_87 = None
    mul_256: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_276, mul_181);  mul_181 = None
    sum_51: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_256, [0, 1]);  mul_256 = None
    sum_52: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_276, [0, 1]);  add_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_12: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_271, torch.float32);  getitem_271 = None
    mul_257: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
    mul_258: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_255, mul_257);  mul_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_967: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_258, [512, 1, 1024, 1, 1]);  mul_258 = None
    permute_1087: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_967, [0, 3, 4, 1, 2]);  view_967 = None
    view_968: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1087, [1, 512, 1024]);  permute_1087 = None
    bmm_207: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1088, view_968);  permute_1088 = None
    bmm_208: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_968, permute_1089);  view_968 = permute_1089 = None
    view_969: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_207, [64, 16, 1, 1024, 1]);  bmm_207 = None
    permute_1090: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_969, [4, 2, 3, 0, 1]);  view_969 = None
    view_970: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_208, [512, 64, 16, 1, 1]);  bmm_208 = None
    permute_1091: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_970, [0, 3, 4, 1, 2]);  view_970 = None
    permute_1092: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1090, [2, 4, 3, 0, 1]);  permute_1090 = None
    squeeze_24: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1092, 4);  permute_1092 = None
    squeeze_25: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_24, 3);  squeeze_24 = None
    permute_1093: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1091, [0, 1, 4, 3, 2]);  permute_1091 = None
    squeeze_26: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1093, 4);  permute_1093 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_971: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_26, [512, 1, 16, 64, 1]);  squeeze_26 = None
    permute_1094: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_971, [2, 0, 4, 1, 3]);  view_971 = None
    view_972: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_1094, [16, 512, 64]);  permute_1094 = None
    bmm_209: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_1095, view_972);  permute_1095 = None
    bmm_210: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_972, permute_1096);  view_972 = permute_1096 = None
    view_973: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_209, [16, 512, 1, 64, 1]);  bmm_209 = None
    permute_1097: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_973, [4, 2, 0, 3, 1]);  view_973 = None
    view_974: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_210, [16, 512, 512, 1, 1]);  bmm_210 = None
    permute_1098: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_974, [1, 3, 0, 4, 2]);  view_974 = None
    permute_1099: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1097, [4, 1, 2, 3, 0]);  permute_1097 = None
    squeeze_27: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1099, 4);  permute_1099 = None
    permute_1100: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_1098, [1, 2, 0, 4, 3]);  permute_1098 = None
    squeeze_28: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_1100, 4);  permute_1100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_13: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_269, torch.float32);  getitem_269 = None
    mul_259: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_260: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_28, mul_259);  squeeze_28 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_261: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_260, alias_27);  mul_260 = None
    sum_53: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_261, [3], True)
    mul_262: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_27, sum_53);  alias_27 = sum_53 = None
    sub_88: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_261, mul_262);  mul_261 = mul_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_263: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_88, 0.125);  sub_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_1: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_263, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_975: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_1, [1, 16, 1023, 512]);  index_put_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_5: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_975, 2, 1, 9223372036854775807);  view_975 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_976: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_5, [1, 16, 512, 1024]);  slice_scatter_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_977: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_976, [1, 16, 512, 1024, 1]);  view_976 = None
    permute_1101: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_977, [1, 2, 4, 0, 3]);  view_977 = None
    view_978: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_1101, [16, 512, 1024]);  permute_1101 = None
    bmm_211: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_1102, view_978);  permute_1102 = None
    bmm_212: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_978, permute_1103);  view_978 = permute_1103 = None
    view_979: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_211, [16, 64, 1, 1024, 1]);  bmm_211 = None
    permute_1104: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_979, [2, 0, 4, 3, 1]);  view_979 = None
    view_980: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_212, [16, 512, 64, 1, 1]);  bmm_212 = None
    permute_1105: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_980, [3, 0, 1, 4, 2]);  view_980 = None
    permute_1106: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1104, [3, 0, 1, 4, 2]);  permute_1104 = None
    squeeze_29: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1106, 4);  permute_1106 = None
    permute_1107: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1105, [2, 0, 1, 4, 3]);  permute_1105 = None
    squeeze_30: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1107, 4);  permute_1107 = None
    sum_54: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_30, [0, 1], True)
    view_981: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_54, [16, 64]);  sum_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_982: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_263, [1, 16, 512, 512, 1]);  mul_263 = None
    permute_1108: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_982, [1, 2, 4, 0, 3]);  view_982 = None
    view_983: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_1108, [16, 512, 512]);  permute_1108 = None
    bmm_213: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_1109, view_983);  permute_1109 = None
    bmm_214: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_983, permute_1110);  view_983 = permute_1110 = None
    view_984: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_213, [16, 64, 1, 512, 1]);  bmm_213 = None
    permute_1111: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_984, [2, 0, 4, 3, 1]);  view_984 = None
    view_985: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_214, [16, 512, 64, 1, 1]);  bmm_214 = None
    permute_1112: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_985, [3, 0, 1, 4, 2]);  view_985 = None
    permute_1113: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1111, [3, 0, 1, 4, 2]);  permute_1111 = None
    squeeze_31: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1113, 4);  permute_1113 = None
    permute_1114: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1112, [2, 0, 1, 4, 3]);  permute_1112 = None
    squeeze_32: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1114, 4);  permute_1114 = None
    sum_55: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_32, [0, 1], True)
    view_986: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_55, [16, 64]);  sum_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_277: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_30, squeeze_32);  squeeze_30 = squeeze_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_987: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_29, [1024, 1, 16, 64, 1]);  squeeze_29 = None
    permute_1115: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_987, [0, 4, 1, 2, 3]);  view_987 = None
    view_988: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_1115, [1, 1024, 1024]);  permute_1115 = None
    bmm_215: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_988);  view_988 = None
    view_989: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_215, [1024, 1, 16, 64, 1]);  bmm_215 = None
    permute_1117: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_989, [4, 1, 2, 3, 0]);  view_989 = None
    permute_1118: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1117, [4, 2, 3, 0, 1]);  permute_1117 = None
    squeeze_33: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1118, 4);  permute_1118 = None
    squeeze_34: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_33, 3);  squeeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_990: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_27, [512, 1, 16, 64, 1]);  squeeze_27 = None
    permute_1119: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_990, [0, 4, 1, 2, 3]);  view_990 = None
    clone_59: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1119, memory_format = torch.contiguous_format);  permute_1119 = None
    view_991: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_59, [1, 512, 1024]);  clone_59 = None
    bmm_216: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1120, view_991)
    bmm_217: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_991, permute_1121);  view_991 = permute_1121 = None
    view_992: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_216, [1024, 1, 16, 64, 1]);  bmm_216 = None
    permute_1122: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_992, [4, 1, 2, 3, 0]);  view_992 = None
    view_993: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_217, [512, 1024, 1, 1, 1]);  bmm_217 = None
    permute_1123: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_993, [0, 2, 3, 4, 1]);  view_993 = None
    permute_1124: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1122, [4, 2, 3, 0, 1]);  permute_1122 = None
    squeeze_35: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1124, 4);  permute_1124 = None
    squeeze_36: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_35, 3);  squeeze_35 = None
    permute_1125: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1123, [0, 1, 4, 2, 3]);  permute_1123 = None
    squeeze_37: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1125, 4);  permute_1125 = None
    squeeze_38: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_37, 3);  squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_278: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_255, squeeze_38);  mul_255 = squeeze_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_994: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_31, [512, 1, 16, 64, 1]);  squeeze_31 = None
    permute_1126: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_994, [0, 4, 1, 2, 3]);  view_994 = None
    view_995: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1126, [1, 512, 1024]);  permute_1126 = None
    bmm_218: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1120, view_995)
    bmm_219: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_995, permute_1128);  view_995 = permute_1128 = None
    view_996: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_218, [1024, 1, 16, 64, 1]);  bmm_218 = None
    permute_1129: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_996, [4, 1, 2, 3, 0]);  view_996 = None
    view_997: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_219, [512, 1024, 1, 1, 1]);  bmm_219 = None
    permute_1130: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_997, [0, 2, 3, 4, 1]);  view_997 = None
    permute_1131: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1129, [4, 2, 3, 0, 1]);  permute_1129 = None
    squeeze_39: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1131, 4);  permute_1131 = None
    squeeze_40: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_39, 3);  squeeze_39 = None
    permute_1132: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1130, [0, 1, 4, 2, 3]);  permute_1130 = None
    squeeze_41: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1132, 4);  permute_1132 = None
    squeeze_42: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_41, 3);  squeeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_279: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_278, squeeze_42);  add_278 = squeeze_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_998: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_277, [512, 1, 16, 64, 1]);  add_277 = None
    permute_1133: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_998, [0, 4, 1, 2, 3]);  view_998 = None
    clone_60: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1133, memory_format = torch.contiguous_format);  permute_1133 = None
    view_999: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_60, [1, 512, 1024]);  clone_60 = None
    bmm_220: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1120, view_999);  permute_1120 = None
    bmm_221: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_999, permute_1135);  view_999 = permute_1135 = None
    view_1000: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_220, [1024, 1, 16, 64, 1]);  bmm_220 = None
    permute_1136: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1000, [4, 1, 2, 3, 0]);  view_1000 = None
    view_1001: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_221, [512, 1024, 1, 1, 1]);  bmm_221 = None
    permute_1137: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1001, [0, 2, 3, 4, 1]);  view_1001 = None
    permute_1138: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1136, [4, 2, 3, 0, 1]);  permute_1136 = None
    squeeze_43: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1138, 4);  permute_1138 = None
    squeeze_44: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_43, 3);  squeeze_43 = None
    permute_1139: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1137, [0, 1, 4, 2, 3]);  permute_1137 = None
    squeeze_45: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1139, 4);  permute_1139 = None
    squeeze_46: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_45, 3);  squeeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_280: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_279, squeeze_46);  add_279 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_265: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_280, primals_344);  primals_344 = None
    mul_266: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_265, 1024)
    sum_56: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_265, [2], True)
    mul_267: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_265, mul_178);  mul_265 = None
    sum_57: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [2], True);  mul_267 = None
    mul_268: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_178, sum_57);  sum_57 = None
    sub_90: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_266, sum_56);  mul_266 = sum_56 = None
    sub_91: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_90, mul_268);  sub_90 = mul_268 = None
    mul_269: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_31, sub_91);  div_31 = sub_91 = None
    mul_270: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_280, mul_178);  mul_178 = None
    sum_58: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_270, [0, 1]);  mul_270 = None
    sum_59: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_280, [0, 1]);  add_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_14: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_265, torch.float32);  getitem_265 = None
    mul_271: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_272: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_269, mul_271);  mul_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1002: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_272, [512, 1024]);  mul_272 = None
    mm_10: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1002, permute_1140);  permute_1140 = None
    permute_1141: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1002, [1, 0])
    mm_11: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1141, view_834);  permute_1141 = view_834 = None
    permute_1142: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_60: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1002, [0], True);  view_1002 = None
    view_1003: "f32[1024]" = torch.ops.aten.reshape.default(sum_60, [1024]);  sum_60 = None
    permute_1143: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1142, [1, 0]);  permute_1142 = None
    view_1004: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_10, [512, 1, 4096]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_15: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_263, torch.float32);  getitem_263 = None
    mul_273: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_274: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1004, mul_273);  view_1004 = mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_276: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_240, 0.5);  add_240 = None
    mul_277: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_833, view_833)
    mul_278: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_277, -0.5);  mul_277 = None
    exp_28: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_278);  mul_278 = None
    mul_279: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_280: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_833, mul_279);  view_833 = mul_279 = None
    add_282: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_276, mul_280);  mul_276 = mul_280 = None
    mul_281: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_274, add_282);  mul_274 = add_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1005: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_281, [512, 4096]);  mul_281 = None
    mm_12: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1005, permute_1144);  permute_1144 = None
    permute_1145: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1005, [1, 0])
    mm_13: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1145, view_832);  permute_1145 = view_832 = None
    permute_1146: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_61: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1005, [0], True);  view_1005 = None
    view_1006: "f32[4096]" = torch.ops.aten.reshape.default(sum_61, [4096]);  sum_61 = None
    permute_1147: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1146, [1, 0]);  permute_1146 = None
    view_1007: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_12, [512, 1, 1024]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_283: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_269, view_1007);  mul_269 = view_1007 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_283: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_283, primals_338);  primals_338 = None
    mul_284: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_283, 1024)
    sum_62: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_283, [2], True)
    mul_285: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_283, mul_173);  mul_283 = None
    sum_63: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_285, [2], True);  mul_285 = None
    mul_286: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_173, sum_63);  sum_63 = None
    sub_93: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_284, sum_62);  mul_284 = sum_62 = None
    sub_94: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_93, mul_286);  sub_93 = mul_286 = None
    mul_287: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_32, sub_94);  div_32 = sub_94 = None
    mul_288: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_283, mul_173);  mul_173 = None
    sum_64: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_288, [0, 1]);  mul_288 = None
    sum_65: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_283, [0, 1]);  add_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_16: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_259, torch.float32);  getitem_259 = None
    mul_289: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_290: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_287, mul_289);  mul_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1008: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_290, [512, 1, 1024, 1, 1]);  mul_290 = None
    permute_1148: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1008, [0, 3, 4, 1, 2]);  view_1008 = None
    view_1009: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1148, [1, 512, 1024]);  permute_1148 = None
    bmm_222: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1149, view_1009);  permute_1149 = None
    bmm_223: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1009, permute_1150);  view_1009 = permute_1150 = None
    view_1010: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_222, [64, 16, 1, 1024, 1]);  bmm_222 = None
    permute_1151: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1010, [4, 2, 3, 0, 1]);  view_1010 = None
    view_1011: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_223, [512, 64, 16, 1, 1]);  bmm_223 = None
    permute_1152: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1011, [0, 3, 4, 1, 2]);  view_1011 = None
    permute_1153: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1151, [2, 4, 3, 0, 1]);  permute_1151 = None
    squeeze_47: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1153, 4);  permute_1153 = None
    squeeze_48: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_47, 3);  squeeze_47 = None
    permute_1154: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1152, [0, 1, 4, 3, 2]);  permute_1152 = None
    squeeze_49: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1154, 4);  permute_1154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1012: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_49, [512, 1, 16, 64, 1]);  squeeze_49 = None
    permute_1155: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1012, [2, 0, 4, 1, 3]);  view_1012 = None
    view_1013: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_1155, [16, 512, 64]);  permute_1155 = None
    bmm_224: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_1156, view_1013);  permute_1156 = None
    bmm_225: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1013, permute_1157);  view_1013 = permute_1157 = None
    view_1014: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_224, [16, 512, 1, 64, 1]);  bmm_224 = None
    permute_1158: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1014, [4, 2, 0, 3, 1]);  view_1014 = None
    view_1015: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_225, [16, 512, 512, 1, 1]);  bmm_225 = None
    permute_1159: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1015, [1, 3, 0, 4, 2]);  view_1015 = None
    permute_1160: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1158, [4, 1, 2, 3, 0]);  permute_1158 = None
    squeeze_50: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1160, 4);  permute_1160 = None
    permute_1161: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_1159, [1, 2, 0, 4, 3]);  permute_1159 = None
    squeeze_51: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_1161, 4);  permute_1161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_17: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_257, torch.float32);  getitem_257 = None
    mul_291: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_292: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_51, mul_291);  squeeze_51 = mul_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_293: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_292, alias_28);  mul_292 = None
    sum_66: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_293, [3], True)
    mul_294: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_28, sum_66);  alias_28 = sum_66 = None
    sub_95: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_293, mul_294);  mul_293 = mul_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_295: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_95, 0.125);  sub_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_2: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_295, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1016: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_2, [1, 16, 1023, 512]);  index_put_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_9: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1016, 2, 1, 9223372036854775807);  view_1016 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1017: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_9, [1, 16, 512, 1024]);  slice_scatter_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1018: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1017, [1, 16, 512, 1024, 1]);  view_1017 = None
    permute_1162: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1018, [1, 2, 4, 0, 3]);  view_1018 = None
    view_1019: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_1162, [16, 512, 1024]);  permute_1162 = None
    bmm_226: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_1163, view_1019);  permute_1163 = None
    bmm_227: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1019, permute_1164);  view_1019 = permute_1164 = None
    view_1020: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_226, [16, 64, 1, 1024, 1]);  bmm_226 = None
    permute_1165: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1020, [2, 0, 4, 3, 1]);  view_1020 = None
    view_1021: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_227, [16, 512, 64, 1, 1]);  bmm_227 = None
    permute_1166: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1021, [3, 0, 1, 4, 2]);  view_1021 = None
    permute_1167: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1165, [3, 0, 1, 4, 2]);  permute_1165 = None
    squeeze_52: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1167, 4);  permute_1167 = None
    permute_1168: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1166, [2, 0, 1, 4, 3]);  permute_1166 = None
    squeeze_53: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1168, 4);  permute_1168 = None
    sum_67: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_53, [0, 1], True)
    view_1022: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_67, [16, 64]);  sum_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1023: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_295, [1, 16, 512, 512, 1]);  mul_295 = None
    permute_1169: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1023, [1, 2, 4, 0, 3]);  view_1023 = None
    view_1024: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_1169, [16, 512, 512]);  permute_1169 = None
    bmm_228: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_1170, view_1024);  permute_1170 = None
    bmm_229: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1024, permute_1171);  view_1024 = permute_1171 = None
    view_1025: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_228, [16, 64, 1, 512, 1]);  bmm_228 = None
    permute_1172: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1025, [2, 0, 4, 3, 1]);  view_1025 = None
    view_1026: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_229, [16, 512, 64, 1, 1]);  bmm_229 = None
    permute_1173: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1026, [3, 0, 1, 4, 2]);  view_1026 = None
    permute_1174: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1172, [3, 0, 1, 4, 2]);  permute_1172 = None
    squeeze_54: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1174, 4);  permute_1174 = None
    permute_1175: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1173, [2, 0, 1, 4, 3]);  permute_1173 = None
    squeeze_55: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1175, 4);  permute_1175 = None
    sum_68: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_55, [0, 1], True)
    view_1027: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_68, [16, 64]);  sum_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_284: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_53, squeeze_55);  squeeze_53 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1028: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_52, [1024, 1, 16, 64, 1]);  squeeze_52 = None
    permute_1176: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1028, [0, 4, 1, 2, 3]);  view_1028 = None
    view_1029: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_1176, [1, 1024, 1024]);  permute_1176 = None
    bmm_230: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1029);  view_1029 = None
    view_1030: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_230, [1024, 1, 16, 64, 1]);  bmm_230 = None
    permute_1178: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1030, [4, 1, 2, 3, 0]);  view_1030 = None
    permute_1179: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1178, [4, 2, 3, 0, 1]);  permute_1178 = None
    squeeze_56: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1179, 4);  permute_1179 = None
    squeeze_57: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_56, 3);  squeeze_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1031: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_50, [512, 1, 16, 64, 1]);  squeeze_50 = None
    permute_1180: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1031, [0, 4, 1, 2, 3]);  view_1031 = None
    clone_65: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1180, memory_format = torch.contiguous_format);  permute_1180 = None
    view_1032: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_65, [1, 512, 1024]);  clone_65 = None
    bmm_231: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1181, view_1032)
    bmm_232: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1032, permute_1182);  view_1032 = permute_1182 = None
    view_1033: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_231, [1024, 1, 16, 64, 1]);  bmm_231 = None
    permute_1183: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1033, [4, 1, 2, 3, 0]);  view_1033 = None
    view_1034: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_232, [512, 1024, 1, 1, 1]);  bmm_232 = None
    permute_1184: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1034, [0, 2, 3, 4, 1]);  view_1034 = None
    permute_1185: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1183, [4, 2, 3, 0, 1]);  permute_1183 = None
    squeeze_58: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1185, 4);  permute_1185 = None
    squeeze_59: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_58, 3);  squeeze_58 = None
    permute_1186: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1184, [0, 1, 4, 2, 3]);  permute_1184 = None
    squeeze_60: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1186, 4);  permute_1186 = None
    squeeze_61: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_60, 3);  squeeze_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_285: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_287, squeeze_61);  mul_287 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1035: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_54, [512, 1, 16, 64, 1]);  squeeze_54 = None
    permute_1187: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1035, [0, 4, 1, 2, 3]);  view_1035 = None
    view_1036: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1187, [1, 512, 1024]);  permute_1187 = None
    bmm_233: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1181, view_1036)
    bmm_234: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1036, permute_1189);  view_1036 = permute_1189 = None
    view_1037: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_233, [1024, 1, 16, 64, 1]);  bmm_233 = None
    permute_1190: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1037, [4, 1, 2, 3, 0]);  view_1037 = None
    view_1038: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_234, [512, 1024, 1, 1, 1]);  bmm_234 = None
    permute_1191: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1038, [0, 2, 3, 4, 1]);  view_1038 = None
    permute_1192: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1190, [4, 2, 3, 0, 1]);  permute_1190 = None
    squeeze_62: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1192, 4);  permute_1192 = None
    squeeze_63: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_62, 3);  squeeze_62 = None
    permute_1193: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1191, [0, 1, 4, 2, 3]);  permute_1191 = None
    squeeze_64: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1193, 4);  permute_1193 = None
    squeeze_65: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_64, 3);  squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_286: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_285, squeeze_65);  add_285 = squeeze_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1039: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_284, [512, 1, 16, 64, 1]);  add_284 = None
    permute_1194: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1039, [0, 4, 1, 2, 3]);  view_1039 = None
    clone_66: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1194, memory_format = torch.contiguous_format);  permute_1194 = None
    view_1040: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_66, [1, 512, 1024]);  clone_66 = None
    bmm_235: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1181, view_1040);  permute_1181 = None
    bmm_236: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1040, permute_1196);  view_1040 = permute_1196 = None
    view_1041: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_235, [1024, 1, 16, 64, 1]);  bmm_235 = None
    permute_1197: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1041, [4, 1, 2, 3, 0]);  view_1041 = None
    view_1042: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_236, [512, 1024, 1, 1, 1]);  bmm_236 = None
    permute_1198: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1042, [0, 2, 3, 4, 1]);  view_1042 = None
    permute_1199: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1197, [4, 2, 3, 0, 1]);  permute_1197 = None
    squeeze_66: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1199, 4);  permute_1199 = None
    squeeze_67: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_66, 3);  squeeze_66 = None
    permute_1200: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1198, [0, 1, 4, 2, 3]);  permute_1198 = None
    squeeze_68: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1200, 4);  permute_1200 = None
    squeeze_69: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_68, 3);  squeeze_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_287: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_286, squeeze_69);  add_286 = squeeze_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_297: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_287, primals_336);  primals_336 = None
    mul_298: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_297, 1024)
    sum_69: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True)
    mul_299: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_297, mul_170);  mul_297 = None
    sum_70: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True);  mul_299 = None
    mul_300: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_170, sum_70);  sum_70 = None
    sub_97: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_298, sum_69);  mul_298 = sum_69 = None
    sub_98: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_97, mul_300);  sub_97 = mul_300 = None
    mul_301: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_33, sub_98);  div_33 = sub_98 = None
    mul_302: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_287, mul_170);  mul_170 = None
    sum_71: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 1]);  mul_302 = None
    sum_72: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_287, [0, 1]);  add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_18: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_253, torch.float32);  getitem_253 = None
    mul_303: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
    mul_304: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_301, mul_303);  mul_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1043: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_304, [512, 1024]);  mul_304 = None
    mm_14: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1043, permute_1201);  permute_1201 = None
    permute_1202: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1043, [1, 0])
    mm_15: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1202, view_796);  permute_1202 = view_796 = None
    permute_1203: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_73: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1043, [0], True);  view_1043 = None
    view_1044: "f32[1024]" = torch.ops.aten.reshape.default(sum_73, [1024]);  sum_73 = None
    permute_1204: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1203, [1, 0]);  permute_1203 = None
    view_1045: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_14, [512, 1, 4096]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_19: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_251, torch.float32);  getitem_251 = None
    mul_305: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_306: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1045, mul_305);  view_1045 = mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_308: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_229, 0.5);  add_229 = None
    mul_309: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_795, view_795)
    mul_310: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_309, -0.5);  mul_309 = None
    exp_29: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_310);  mul_310 = None
    mul_311: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_312: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_795, mul_311);  view_795 = mul_311 = None
    add_289: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_308, mul_312);  mul_308 = mul_312 = None
    mul_313: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_306, add_289);  mul_306 = add_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1046: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_313, [512, 4096]);  mul_313 = None
    mm_16: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1046, permute_1205);  permute_1205 = None
    permute_1206: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1046, [1, 0])
    mm_17: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1206, view_794);  permute_1206 = view_794 = None
    permute_1207: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_74: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1046, [0], True);  view_1046 = None
    view_1047: "f32[4096]" = torch.ops.aten.reshape.default(sum_74, [4096]);  sum_74 = None
    permute_1208: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1207, [1, 0]);  permute_1207 = None
    view_1048: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_16, [512, 1, 1024]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_290: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_301, view_1048);  mul_301 = view_1048 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_315: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_290, primals_330);  primals_330 = None
    mul_316: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_315, 1024)
    sum_75: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [2], True)
    mul_317: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_315, mul_165);  mul_315 = None
    sum_76: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_317, [2], True);  mul_317 = None
    mul_318: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_165, sum_76);  sum_76 = None
    sub_100: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_316, sum_75);  mul_316 = sum_75 = None
    sub_101: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_100, mul_318);  sub_100 = mul_318 = None
    mul_319: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_34, sub_101);  div_34 = sub_101 = None
    mul_320: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_290, mul_165);  mul_165 = None
    sum_77: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_320, [0, 1]);  mul_320 = None
    sum_78: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_290, [0, 1]);  add_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_20: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_247, torch.float32);  getitem_247 = None
    mul_321: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_322: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_319, mul_321);  mul_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1049: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_322, [512, 1, 1024, 1, 1]);  mul_322 = None
    permute_1209: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1049, [0, 3, 4, 1, 2]);  view_1049 = None
    view_1050: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1209, [1, 512, 1024]);  permute_1209 = None
    bmm_237: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1210, view_1050);  permute_1210 = None
    bmm_238: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1050, permute_1211);  view_1050 = permute_1211 = None
    view_1051: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_237, [64, 16, 1, 1024, 1]);  bmm_237 = None
    permute_1212: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1051, [4, 2, 3, 0, 1]);  view_1051 = None
    view_1052: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_238, [512, 64, 16, 1, 1]);  bmm_238 = None
    permute_1213: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1052, [0, 3, 4, 1, 2]);  view_1052 = None
    permute_1214: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1212, [2, 4, 3, 0, 1]);  permute_1212 = None
    squeeze_70: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1214, 4);  permute_1214 = None
    squeeze_71: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_70, 3);  squeeze_70 = None
    permute_1215: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1213, [0, 1, 4, 3, 2]);  permute_1213 = None
    squeeze_72: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1215, 4);  permute_1215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1053: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_72, [512, 1, 16, 64, 1]);  squeeze_72 = None
    permute_1216: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1053, [2, 0, 4, 1, 3]);  view_1053 = None
    view_1054: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_1216, [16, 512, 64]);  permute_1216 = None
    bmm_239: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_1217, view_1054);  permute_1217 = None
    bmm_240: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1054, permute_1218);  view_1054 = permute_1218 = None
    view_1055: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_239, [16, 512, 1, 64, 1]);  bmm_239 = None
    permute_1219: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1055, [4, 2, 0, 3, 1]);  view_1055 = None
    view_1056: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_240, [16, 512, 512, 1, 1]);  bmm_240 = None
    permute_1220: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1056, [1, 3, 0, 4, 2]);  view_1056 = None
    permute_1221: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1219, [4, 1, 2, 3, 0]);  permute_1219 = None
    squeeze_73: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1221, 4);  permute_1221 = None
    permute_1222: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_1220, [1, 2, 0, 4, 3]);  permute_1220 = None
    squeeze_74: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_1222, 4);  permute_1222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_21: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_245, torch.float32);  getitem_245 = None
    mul_323: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
    mul_324: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_74, mul_323);  squeeze_74 = mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_325: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_324, alias_29);  mul_324 = None
    sum_79: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [3], True)
    mul_326: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_29, sum_79);  alias_29 = sum_79 = None
    sub_102: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_325, mul_326);  mul_325 = mul_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_327: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_102, 0.125);  sub_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_3: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_327, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1057: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_3, [1, 16, 1023, 512]);  index_put_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_13: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1057, 2, 1, 9223372036854775807);  view_1057 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1058: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_13, [1, 16, 512, 1024]);  slice_scatter_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1059: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1058, [1, 16, 512, 1024, 1]);  view_1058 = None
    permute_1223: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1059, [1, 2, 4, 0, 3]);  view_1059 = None
    view_1060: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_1223, [16, 512, 1024]);  permute_1223 = None
    bmm_241: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_1224, view_1060);  permute_1224 = None
    bmm_242: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1060, permute_1225);  view_1060 = permute_1225 = None
    view_1061: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_241, [16, 64, 1, 1024, 1]);  bmm_241 = None
    permute_1226: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1061, [2, 0, 4, 3, 1]);  view_1061 = None
    view_1062: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_242, [16, 512, 64, 1, 1]);  bmm_242 = None
    permute_1227: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1062, [3, 0, 1, 4, 2]);  view_1062 = None
    permute_1228: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1226, [3, 0, 1, 4, 2]);  permute_1226 = None
    squeeze_75: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1228, 4);  permute_1228 = None
    permute_1229: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1227, [2, 0, 1, 4, 3]);  permute_1227 = None
    squeeze_76: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1229, 4);  permute_1229 = None
    sum_80: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_76, [0, 1], True)
    view_1063: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_80, [16, 64]);  sum_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1064: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_327, [1, 16, 512, 512, 1]);  mul_327 = None
    permute_1230: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1064, [1, 2, 4, 0, 3]);  view_1064 = None
    view_1065: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_1230, [16, 512, 512]);  permute_1230 = None
    bmm_243: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_1231, view_1065);  permute_1231 = None
    bmm_244: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1065, permute_1232);  view_1065 = permute_1232 = None
    view_1066: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_243, [16, 64, 1, 512, 1]);  bmm_243 = None
    permute_1233: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1066, [2, 0, 4, 3, 1]);  view_1066 = None
    view_1067: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_244, [16, 512, 64, 1, 1]);  bmm_244 = None
    permute_1234: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1067, [3, 0, 1, 4, 2]);  view_1067 = None
    permute_1235: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1233, [3, 0, 1, 4, 2]);  permute_1233 = None
    squeeze_77: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1235, 4);  permute_1235 = None
    permute_1236: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1234, [2, 0, 1, 4, 3]);  permute_1234 = None
    squeeze_78: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1236, 4);  permute_1236 = None
    sum_81: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_78, [0, 1], True)
    view_1068: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_81, [16, 64]);  sum_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_291: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_76, squeeze_78);  squeeze_76 = squeeze_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1069: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_75, [1024, 1, 16, 64, 1]);  squeeze_75 = None
    permute_1237: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1069, [0, 4, 1, 2, 3]);  view_1069 = None
    view_1070: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_1237, [1, 1024, 1024]);  permute_1237 = None
    bmm_245: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1070);  view_1070 = None
    view_1071: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_245, [1024, 1, 16, 64, 1]);  bmm_245 = None
    permute_1239: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1071, [4, 1, 2, 3, 0]);  view_1071 = None
    permute_1240: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1239, [4, 2, 3, 0, 1]);  permute_1239 = None
    squeeze_79: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1240, 4);  permute_1240 = None
    squeeze_80: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_79, 3);  squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1072: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_73, [512, 1, 16, 64, 1]);  squeeze_73 = None
    permute_1241: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1072, [0, 4, 1, 2, 3]);  view_1072 = None
    clone_71: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1241, memory_format = torch.contiguous_format);  permute_1241 = None
    view_1073: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_71, [1, 512, 1024]);  clone_71 = None
    bmm_246: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1242, view_1073)
    bmm_247: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1073, permute_1243);  view_1073 = permute_1243 = None
    view_1074: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_246, [1024, 1, 16, 64, 1]);  bmm_246 = None
    permute_1244: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1074, [4, 1, 2, 3, 0]);  view_1074 = None
    view_1075: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_247, [512, 1024, 1, 1, 1]);  bmm_247 = None
    permute_1245: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1075, [0, 2, 3, 4, 1]);  view_1075 = None
    permute_1246: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1244, [4, 2, 3, 0, 1]);  permute_1244 = None
    squeeze_81: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1246, 4);  permute_1246 = None
    squeeze_82: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_81, 3);  squeeze_81 = None
    permute_1247: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1245, [0, 1, 4, 2, 3]);  permute_1245 = None
    squeeze_83: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1247, 4);  permute_1247 = None
    squeeze_84: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_83, 3);  squeeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_292: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_319, squeeze_84);  mul_319 = squeeze_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1076: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_77, [512, 1, 16, 64, 1]);  squeeze_77 = None
    permute_1248: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1076, [0, 4, 1, 2, 3]);  view_1076 = None
    view_1077: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1248, [1, 512, 1024]);  permute_1248 = None
    bmm_248: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1242, view_1077)
    bmm_249: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1077, permute_1250);  view_1077 = permute_1250 = None
    view_1078: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_248, [1024, 1, 16, 64, 1]);  bmm_248 = None
    permute_1251: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1078, [4, 1, 2, 3, 0]);  view_1078 = None
    view_1079: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_249, [512, 1024, 1, 1, 1]);  bmm_249 = None
    permute_1252: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1079, [0, 2, 3, 4, 1]);  view_1079 = None
    permute_1253: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1251, [4, 2, 3, 0, 1]);  permute_1251 = None
    squeeze_85: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1253, 4);  permute_1253 = None
    squeeze_86: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_85, 3);  squeeze_85 = None
    permute_1254: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1252, [0, 1, 4, 2, 3]);  permute_1252 = None
    squeeze_87: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1254, 4);  permute_1254 = None
    squeeze_88: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_87, 3);  squeeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_293: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_292, squeeze_88);  add_292 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1080: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_291, [512, 1, 16, 64, 1]);  add_291 = None
    permute_1255: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1080, [0, 4, 1, 2, 3]);  view_1080 = None
    clone_72: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1255, memory_format = torch.contiguous_format);  permute_1255 = None
    view_1081: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_72, [1, 512, 1024]);  clone_72 = None
    bmm_250: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1242, view_1081);  permute_1242 = None
    bmm_251: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1081, permute_1257);  view_1081 = permute_1257 = None
    view_1082: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_250, [1024, 1, 16, 64, 1]);  bmm_250 = None
    permute_1258: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1082, [4, 1, 2, 3, 0]);  view_1082 = None
    view_1083: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_251, [512, 1024, 1, 1, 1]);  bmm_251 = None
    permute_1259: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1083, [0, 2, 3, 4, 1]);  view_1083 = None
    permute_1260: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1258, [4, 2, 3, 0, 1]);  permute_1258 = None
    squeeze_89: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1260, 4);  permute_1260 = None
    squeeze_90: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_89, 3);  squeeze_89 = None
    permute_1261: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1259, [0, 1, 4, 2, 3]);  permute_1259 = None
    squeeze_91: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1261, 4);  permute_1261 = None
    squeeze_92: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_91, 3);  squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_294: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_293, squeeze_92);  add_293 = squeeze_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_329: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_294, primals_328);  primals_328 = None
    mul_330: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_329, 1024)
    sum_82: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True)
    mul_331: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_329, mul_162);  mul_329 = None
    sum_83: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_331, [2], True);  mul_331 = None
    mul_332: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_162, sum_83);  sum_83 = None
    sub_104: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_330, sum_82);  mul_330 = sum_82 = None
    sub_105: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_104, mul_332);  sub_104 = mul_332 = None
    mul_333: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_35, sub_105);  div_35 = sub_105 = None
    mul_334: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_294, mul_162);  mul_162 = None
    sum_84: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_334, [0, 1]);  mul_334 = None
    sum_85: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_294, [0, 1]);  add_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_22: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_241, torch.float32);  getitem_241 = None
    mul_335: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_336: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_333, mul_335);  mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1084: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_336, [512, 1024]);  mul_336 = None
    mm_18: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1084, permute_1262);  permute_1262 = None
    permute_1263: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1084, [1, 0])
    mm_19: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1263, view_758);  permute_1263 = view_758 = None
    permute_1264: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_86: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1084, [0], True);  view_1084 = None
    view_1085: "f32[1024]" = torch.ops.aten.reshape.default(sum_86, [1024]);  sum_86 = None
    permute_1265: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1264, [1, 0]);  permute_1264 = None
    view_1086: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_18, [512, 1, 4096]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_23: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_239, torch.float32);  getitem_239 = None
    mul_337: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_338: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1086, mul_337);  view_1086 = mul_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_340: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_218, 0.5);  add_218 = None
    mul_341: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_757, view_757)
    mul_342: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_341, -0.5);  mul_341 = None
    exp_30: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_342);  mul_342 = None
    mul_343: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_344: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_757, mul_343);  view_757 = mul_343 = None
    add_296: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_340, mul_344);  mul_340 = mul_344 = None
    mul_345: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_338, add_296);  mul_338 = add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1087: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_345, [512, 4096]);  mul_345 = None
    mm_20: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1087, permute_1266);  permute_1266 = None
    permute_1267: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1087, [1, 0])
    mm_21: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1267, view_756);  permute_1267 = view_756 = None
    permute_1268: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_87: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1087, [0], True);  view_1087 = None
    view_1088: "f32[4096]" = torch.ops.aten.reshape.default(sum_87, [4096]);  sum_87 = None
    permute_1269: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1268, [1, 0]);  permute_1268 = None
    view_1089: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_20, [512, 1, 1024]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_297: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_333, view_1089);  mul_333 = view_1089 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_347: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_297, primals_322);  primals_322 = None
    mul_348: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_347, 1024)
    sum_88: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True)
    mul_349: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_347, mul_157);  mul_347 = None
    sum_89: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True);  mul_349 = None
    mul_350: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_157, sum_89);  sum_89 = None
    sub_107: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_348, sum_88);  mul_348 = sum_88 = None
    sub_108: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_107, mul_350);  sub_107 = mul_350 = None
    mul_351: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_36, sub_108);  div_36 = sub_108 = None
    mul_352: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_297, mul_157);  mul_157 = None
    sum_90: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 1]);  mul_352 = None
    sum_91: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_297, [0, 1]);  add_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_24: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_235, torch.float32);  getitem_235 = None
    mul_353: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
    mul_354: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_351, mul_353);  mul_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1090: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_354, [512, 1, 1024, 1, 1]);  mul_354 = None
    permute_1270: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1090, [0, 3, 4, 1, 2]);  view_1090 = None
    view_1091: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1270, [1, 512, 1024]);  permute_1270 = None
    bmm_252: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1271, view_1091);  permute_1271 = None
    bmm_253: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1091, permute_1272);  view_1091 = permute_1272 = None
    view_1092: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_252, [64, 16, 1, 1024, 1]);  bmm_252 = None
    permute_1273: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1092, [4, 2, 3, 0, 1]);  view_1092 = None
    view_1093: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_253, [512, 64, 16, 1, 1]);  bmm_253 = None
    permute_1274: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1093, [0, 3, 4, 1, 2]);  view_1093 = None
    permute_1275: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1273, [2, 4, 3, 0, 1]);  permute_1273 = None
    squeeze_93: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1275, 4);  permute_1275 = None
    squeeze_94: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_93, 3);  squeeze_93 = None
    permute_1276: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1274, [0, 1, 4, 3, 2]);  permute_1274 = None
    squeeze_95: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1276, 4);  permute_1276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1094: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_95, [512, 1, 16, 64, 1]);  squeeze_95 = None
    permute_1277: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1094, [2, 0, 4, 1, 3]);  view_1094 = None
    view_1095: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_1277, [16, 512, 64]);  permute_1277 = None
    bmm_254: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_1278, view_1095);  permute_1278 = None
    bmm_255: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1095, permute_1279);  view_1095 = permute_1279 = None
    view_1096: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_254, [16, 512, 1, 64, 1]);  bmm_254 = None
    permute_1280: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1096, [4, 2, 0, 3, 1]);  view_1096 = None
    view_1097: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_255, [16, 512, 512, 1, 1]);  bmm_255 = None
    permute_1281: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1097, [1, 3, 0, 4, 2]);  view_1097 = None
    permute_1282: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1280, [4, 1, 2, 3, 0]);  permute_1280 = None
    squeeze_96: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1282, 4);  permute_1282 = None
    permute_1283: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_1281, [1, 2, 0, 4, 3]);  permute_1281 = None
    squeeze_97: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_1283, 4);  permute_1283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_25: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_233, torch.float32);  getitem_233 = None
    mul_355: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_356: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_97, mul_355);  squeeze_97 = mul_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_357: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_356, alias_30);  mul_356 = None
    sum_92: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [3], True)
    mul_358: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_30, sum_92);  alias_30 = sum_92 = None
    sub_109: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_357, mul_358);  mul_357 = mul_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_359: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_109, 0.125);  sub_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_4: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_359, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1098: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_4, [1, 16, 1023, 512]);  index_put_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_17: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1098, 2, 1, 9223372036854775807);  view_1098 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1099: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_17, [1, 16, 512, 1024]);  slice_scatter_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1100: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1099, [1, 16, 512, 1024, 1]);  view_1099 = None
    permute_1284: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1100, [1, 2, 4, 0, 3]);  view_1100 = None
    view_1101: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_1284, [16, 512, 1024]);  permute_1284 = None
    bmm_256: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_1285, view_1101);  permute_1285 = None
    bmm_257: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1101, permute_1286);  view_1101 = permute_1286 = None
    view_1102: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_256, [16, 64, 1, 1024, 1]);  bmm_256 = None
    permute_1287: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1102, [2, 0, 4, 3, 1]);  view_1102 = None
    view_1103: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_257, [16, 512, 64, 1, 1]);  bmm_257 = None
    permute_1288: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1103, [3, 0, 1, 4, 2]);  view_1103 = None
    permute_1289: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1287, [3, 0, 1, 4, 2]);  permute_1287 = None
    squeeze_98: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1289, 4);  permute_1289 = None
    permute_1290: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1288, [2, 0, 1, 4, 3]);  permute_1288 = None
    squeeze_99: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1290, 4);  permute_1290 = None
    sum_93: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_99, [0, 1], True)
    view_1104: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_93, [16, 64]);  sum_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1105: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_359, [1, 16, 512, 512, 1]);  mul_359 = None
    permute_1291: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1105, [1, 2, 4, 0, 3]);  view_1105 = None
    view_1106: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_1291, [16, 512, 512]);  permute_1291 = None
    bmm_258: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_1292, view_1106);  permute_1292 = None
    bmm_259: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1106, permute_1293);  view_1106 = permute_1293 = None
    view_1107: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_258, [16, 64, 1, 512, 1]);  bmm_258 = None
    permute_1294: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1107, [2, 0, 4, 3, 1]);  view_1107 = None
    view_1108: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_259, [16, 512, 64, 1, 1]);  bmm_259 = None
    permute_1295: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1108, [3, 0, 1, 4, 2]);  view_1108 = None
    permute_1296: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1294, [3, 0, 1, 4, 2]);  permute_1294 = None
    squeeze_100: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1296, 4);  permute_1296 = None
    permute_1297: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1295, [2, 0, 1, 4, 3]);  permute_1295 = None
    squeeze_101: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1297, 4);  permute_1297 = None
    sum_94: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_101, [0, 1], True)
    view_1109: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_94, [16, 64]);  sum_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_298: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_99, squeeze_101);  squeeze_99 = squeeze_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1110: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_98, [1024, 1, 16, 64, 1]);  squeeze_98 = None
    permute_1298: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1110, [0, 4, 1, 2, 3]);  view_1110 = None
    view_1111: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_1298, [1, 1024, 1024]);  permute_1298 = None
    bmm_260: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1111);  view_1111 = None
    view_1112: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_260, [1024, 1, 16, 64, 1]);  bmm_260 = None
    permute_1300: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1112, [4, 1, 2, 3, 0]);  view_1112 = None
    permute_1301: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1300, [4, 2, 3, 0, 1]);  permute_1300 = None
    squeeze_102: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1301, 4);  permute_1301 = None
    squeeze_103: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_102, 3);  squeeze_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1113: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_96, [512, 1, 16, 64, 1]);  squeeze_96 = None
    permute_1302: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1113, [0, 4, 1, 2, 3]);  view_1113 = None
    clone_77: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1302, memory_format = torch.contiguous_format);  permute_1302 = None
    view_1114: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_77, [1, 512, 1024]);  clone_77 = None
    bmm_261: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1303, view_1114)
    bmm_262: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1114, permute_1304);  view_1114 = permute_1304 = None
    view_1115: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_261, [1024, 1, 16, 64, 1]);  bmm_261 = None
    permute_1305: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1115, [4, 1, 2, 3, 0]);  view_1115 = None
    view_1116: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_262, [512, 1024, 1, 1, 1]);  bmm_262 = None
    permute_1306: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1116, [0, 2, 3, 4, 1]);  view_1116 = None
    permute_1307: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1305, [4, 2, 3, 0, 1]);  permute_1305 = None
    squeeze_104: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1307, 4);  permute_1307 = None
    squeeze_105: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_104, 3);  squeeze_104 = None
    permute_1308: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1306, [0, 1, 4, 2, 3]);  permute_1306 = None
    squeeze_106: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1308, 4);  permute_1308 = None
    squeeze_107: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_106, 3);  squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_299: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_351, squeeze_107);  mul_351 = squeeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1117: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_100, [512, 1, 16, 64, 1]);  squeeze_100 = None
    permute_1309: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1117, [0, 4, 1, 2, 3]);  view_1117 = None
    view_1118: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1309, [1, 512, 1024]);  permute_1309 = None
    bmm_263: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1303, view_1118)
    bmm_264: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1118, permute_1311);  view_1118 = permute_1311 = None
    view_1119: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_263, [1024, 1, 16, 64, 1]);  bmm_263 = None
    permute_1312: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1119, [4, 1, 2, 3, 0]);  view_1119 = None
    view_1120: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_264, [512, 1024, 1, 1, 1]);  bmm_264 = None
    permute_1313: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1120, [0, 2, 3, 4, 1]);  view_1120 = None
    permute_1314: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1312, [4, 2, 3, 0, 1]);  permute_1312 = None
    squeeze_108: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1314, 4);  permute_1314 = None
    squeeze_109: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_108, 3);  squeeze_108 = None
    permute_1315: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1313, [0, 1, 4, 2, 3]);  permute_1313 = None
    squeeze_110: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1315, 4);  permute_1315 = None
    squeeze_111: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_110, 3);  squeeze_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_300: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_299, squeeze_111);  add_299 = squeeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1121: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_298, [512, 1, 16, 64, 1]);  add_298 = None
    permute_1316: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1121, [0, 4, 1, 2, 3]);  view_1121 = None
    clone_78: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1316, memory_format = torch.contiguous_format);  permute_1316 = None
    view_1122: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_78, [1, 512, 1024]);  clone_78 = None
    bmm_265: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1303, view_1122);  permute_1303 = None
    bmm_266: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1122, permute_1318);  view_1122 = permute_1318 = None
    view_1123: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_265, [1024, 1, 16, 64, 1]);  bmm_265 = None
    permute_1319: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1123, [4, 1, 2, 3, 0]);  view_1123 = None
    view_1124: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_266, [512, 1024, 1, 1, 1]);  bmm_266 = None
    permute_1320: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1124, [0, 2, 3, 4, 1]);  view_1124 = None
    permute_1321: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1319, [4, 2, 3, 0, 1]);  permute_1319 = None
    squeeze_112: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1321, 4);  permute_1321 = None
    squeeze_113: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_112, 3);  squeeze_112 = None
    permute_1322: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1320, [0, 1, 4, 2, 3]);  permute_1320 = None
    squeeze_114: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1322, 4);  permute_1322 = None
    squeeze_115: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_114, 3);  squeeze_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_301: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_300, squeeze_115);  add_300 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_361: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_301, primals_320);  primals_320 = None
    mul_362: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_361, 1024)
    sum_95: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_361, [2], True)
    mul_363: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_361, mul_154);  mul_361 = None
    sum_96: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_363, [2], True);  mul_363 = None
    mul_364: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_154, sum_96);  sum_96 = None
    sub_111: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_362, sum_95);  mul_362 = sum_95 = None
    sub_112: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_111, mul_364);  sub_111 = mul_364 = None
    mul_365: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_37, sub_112);  div_37 = sub_112 = None
    mul_366: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_301, mul_154);  mul_154 = None
    sum_97: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_366, [0, 1]);  mul_366 = None
    sum_98: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_301, [0, 1]);  add_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_26: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_229, torch.float32);  getitem_229 = None
    mul_367: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_368: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_365, mul_367);  mul_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1125: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_368, [512, 1024]);  mul_368 = None
    mm_22: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1125, permute_1323);  permute_1323 = None
    permute_1324: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1125, [1, 0])
    mm_23: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1324, view_720);  permute_1324 = view_720 = None
    permute_1325: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_99: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1125, [0], True);  view_1125 = None
    view_1126: "f32[1024]" = torch.ops.aten.reshape.default(sum_99, [1024]);  sum_99 = None
    permute_1326: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1325, [1, 0]);  permute_1325 = None
    view_1127: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_22, [512, 1, 4096]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_27: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_227, torch.float32);  getitem_227 = None
    mul_369: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
    mul_370: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1127, mul_369);  view_1127 = mul_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_372: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_207, 0.5);  add_207 = None
    mul_373: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_719, view_719)
    mul_374: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_373, -0.5);  mul_373 = None
    exp_31: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_374);  mul_374 = None
    mul_375: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_376: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_719, mul_375);  view_719 = mul_375 = None
    add_303: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_372, mul_376);  mul_372 = mul_376 = None
    mul_377: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_370, add_303);  mul_370 = add_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1128: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_377, [512, 4096]);  mul_377 = None
    mm_24: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1128, permute_1327);  permute_1327 = None
    permute_1328: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1128, [1, 0])
    mm_25: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1328, view_718);  permute_1328 = view_718 = None
    permute_1329: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_100: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1128, [0], True);  view_1128 = None
    view_1129: "f32[4096]" = torch.ops.aten.reshape.default(sum_100, [4096]);  sum_100 = None
    permute_1330: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1329, [1, 0]);  permute_1329 = None
    view_1130: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_24, [512, 1, 1024]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_304: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_365, view_1130);  mul_365 = view_1130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_379: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_304, primals_314);  primals_314 = None
    mul_380: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_379, 1024)
    sum_101: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [2], True)
    mul_381: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_379, mul_149);  mul_379 = None
    sum_102: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [2], True);  mul_381 = None
    mul_382: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_149, sum_102);  sum_102 = None
    sub_114: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_380, sum_101);  mul_380 = sum_101 = None
    sub_115: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_114, mul_382);  sub_114 = mul_382 = None
    mul_383: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_38, sub_115);  div_38 = sub_115 = None
    mul_384: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_304, mul_149);  mul_149 = None
    sum_103: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_384, [0, 1]);  mul_384 = None
    sum_104: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_304, [0, 1]);  add_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_28: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_223, torch.float32);  getitem_223 = None
    mul_385: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
    mul_386: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_383, mul_385);  mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1131: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_386, [512, 1, 1024, 1, 1]);  mul_386 = None
    permute_1331: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1131, [0, 3, 4, 1, 2]);  view_1131 = None
    view_1132: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1331, [1, 512, 1024]);  permute_1331 = None
    bmm_267: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1332, view_1132);  permute_1332 = None
    bmm_268: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1132, permute_1333);  view_1132 = permute_1333 = None
    view_1133: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_267, [64, 16, 1, 1024, 1]);  bmm_267 = None
    permute_1334: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1133, [4, 2, 3, 0, 1]);  view_1133 = None
    view_1134: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_268, [512, 64, 16, 1, 1]);  bmm_268 = None
    permute_1335: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1134, [0, 3, 4, 1, 2]);  view_1134 = None
    permute_1336: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1334, [2, 4, 3, 0, 1]);  permute_1334 = None
    squeeze_116: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1336, 4);  permute_1336 = None
    squeeze_117: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_116, 3);  squeeze_116 = None
    permute_1337: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1335, [0, 1, 4, 3, 2]);  permute_1335 = None
    squeeze_118: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1337, 4);  permute_1337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1135: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_118, [512, 1, 16, 64, 1]);  squeeze_118 = None
    permute_1338: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1135, [2, 0, 4, 1, 3]);  view_1135 = None
    view_1136: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_1338, [16, 512, 64]);  permute_1338 = None
    bmm_269: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_1339, view_1136);  permute_1339 = None
    bmm_270: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1136, permute_1340);  view_1136 = permute_1340 = None
    view_1137: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_269, [16, 512, 1, 64, 1]);  bmm_269 = None
    permute_1341: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1137, [4, 2, 0, 3, 1]);  view_1137 = None
    view_1138: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_270, [16, 512, 512, 1, 1]);  bmm_270 = None
    permute_1342: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1138, [1, 3, 0, 4, 2]);  view_1138 = None
    permute_1343: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1341, [4, 1, 2, 3, 0]);  permute_1341 = None
    squeeze_119: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1343, 4);  permute_1343 = None
    permute_1344: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_1342, [1, 2, 0, 4, 3]);  permute_1342 = None
    squeeze_120: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_1344, 4);  permute_1344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_29: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_221, torch.float32);  getitem_221 = None
    mul_387: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_388: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_120, mul_387);  squeeze_120 = mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_389: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_388, alias_31);  mul_388 = None
    sum_105: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_389, [3], True)
    mul_390: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_31, sum_105);  alias_31 = sum_105 = None
    sub_116: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_391: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_116, 0.125);  sub_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_5: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_391, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1139: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_5, [1, 16, 1023, 512]);  index_put_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_21: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1139, 2, 1, 9223372036854775807);  view_1139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1140: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_21, [1, 16, 512, 1024]);  slice_scatter_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1141: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1140, [1, 16, 512, 1024, 1]);  view_1140 = None
    permute_1345: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1141, [1, 2, 4, 0, 3]);  view_1141 = None
    view_1142: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_1345, [16, 512, 1024]);  permute_1345 = None
    bmm_271: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_1346, view_1142);  permute_1346 = None
    bmm_272: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1142, permute_1347);  view_1142 = permute_1347 = None
    view_1143: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_271, [16, 64, 1, 1024, 1]);  bmm_271 = None
    permute_1348: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1143, [2, 0, 4, 3, 1]);  view_1143 = None
    view_1144: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_272, [16, 512, 64, 1, 1]);  bmm_272 = None
    permute_1349: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1144, [3, 0, 1, 4, 2]);  view_1144 = None
    permute_1350: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1348, [3, 0, 1, 4, 2]);  permute_1348 = None
    squeeze_121: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1350, 4);  permute_1350 = None
    permute_1351: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1349, [2, 0, 1, 4, 3]);  permute_1349 = None
    squeeze_122: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1351, 4);  permute_1351 = None
    sum_106: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_122, [0, 1], True)
    view_1145: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_106, [16, 64]);  sum_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1146: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_391, [1, 16, 512, 512, 1]);  mul_391 = None
    permute_1352: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1146, [1, 2, 4, 0, 3]);  view_1146 = None
    view_1147: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_1352, [16, 512, 512]);  permute_1352 = None
    bmm_273: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_1353, view_1147);  permute_1353 = None
    bmm_274: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1147, permute_1354);  view_1147 = permute_1354 = None
    view_1148: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_273, [16, 64, 1, 512, 1]);  bmm_273 = None
    permute_1355: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1148, [2, 0, 4, 3, 1]);  view_1148 = None
    view_1149: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_274, [16, 512, 64, 1, 1]);  bmm_274 = None
    permute_1356: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1149, [3, 0, 1, 4, 2]);  view_1149 = None
    permute_1357: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1355, [3, 0, 1, 4, 2]);  permute_1355 = None
    squeeze_123: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1357, 4);  permute_1357 = None
    permute_1358: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1356, [2, 0, 1, 4, 3]);  permute_1356 = None
    squeeze_124: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1358, 4);  permute_1358 = None
    sum_107: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_124, [0, 1], True)
    view_1150: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_107, [16, 64]);  sum_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_305: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_122, squeeze_124);  squeeze_122 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1151: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_121, [1024, 1, 16, 64, 1]);  squeeze_121 = None
    permute_1359: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1151, [0, 4, 1, 2, 3]);  view_1151 = None
    view_1152: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_1359, [1, 1024, 1024]);  permute_1359 = None
    bmm_275: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1152);  view_1152 = None
    view_1153: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_275, [1024, 1, 16, 64, 1]);  bmm_275 = None
    permute_1361: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1153, [4, 1, 2, 3, 0]);  view_1153 = None
    permute_1362: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1361, [4, 2, 3, 0, 1]);  permute_1361 = None
    squeeze_125: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1362, 4);  permute_1362 = None
    squeeze_126: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_125, 3);  squeeze_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1154: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_119, [512, 1, 16, 64, 1]);  squeeze_119 = None
    permute_1363: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1154, [0, 4, 1, 2, 3]);  view_1154 = None
    clone_83: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1363, memory_format = torch.contiguous_format);  permute_1363 = None
    view_1155: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_83, [1, 512, 1024]);  clone_83 = None
    bmm_276: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1364, view_1155)
    bmm_277: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1155, permute_1365);  view_1155 = permute_1365 = None
    view_1156: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_276, [1024, 1, 16, 64, 1]);  bmm_276 = None
    permute_1366: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1156, [4, 1, 2, 3, 0]);  view_1156 = None
    view_1157: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_277, [512, 1024, 1, 1, 1]);  bmm_277 = None
    permute_1367: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1157, [0, 2, 3, 4, 1]);  view_1157 = None
    permute_1368: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1366, [4, 2, 3, 0, 1]);  permute_1366 = None
    squeeze_127: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1368, 4);  permute_1368 = None
    squeeze_128: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_127, 3);  squeeze_127 = None
    permute_1369: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1367, [0, 1, 4, 2, 3]);  permute_1367 = None
    squeeze_129: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1369, 4);  permute_1369 = None
    squeeze_130: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_129, 3);  squeeze_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_306: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_383, squeeze_130);  mul_383 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1158: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_123, [512, 1, 16, 64, 1]);  squeeze_123 = None
    permute_1370: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1158, [0, 4, 1, 2, 3]);  view_1158 = None
    view_1159: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1370, [1, 512, 1024]);  permute_1370 = None
    bmm_278: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1364, view_1159)
    bmm_279: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1159, permute_1372);  view_1159 = permute_1372 = None
    view_1160: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_278, [1024, 1, 16, 64, 1]);  bmm_278 = None
    permute_1373: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1160, [4, 1, 2, 3, 0]);  view_1160 = None
    view_1161: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_279, [512, 1024, 1, 1, 1]);  bmm_279 = None
    permute_1374: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1161, [0, 2, 3, 4, 1]);  view_1161 = None
    permute_1375: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1373, [4, 2, 3, 0, 1]);  permute_1373 = None
    squeeze_131: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1375, 4);  permute_1375 = None
    squeeze_132: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_131, 3);  squeeze_131 = None
    permute_1376: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1374, [0, 1, 4, 2, 3]);  permute_1374 = None
    squeeze_133: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1376, 4);  permute_1376 = None
    squeeze_134: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_133, 3);  squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_307: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_306, squeeze_134);  add_306 = squeeze_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1162: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_305, [512, 1, 16, 64, 1]);  add_305 = None
    permute_1377: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1162, [0, 4, 1, 2, 3]);  view_1162 = None
    clone_84: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1377, memory_format = torch.contiguous_format);  permute_1377 = None
    view_1163: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_84, [1, 512, 1024]);  clone_84 = None
    bmm_280: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1364, view_1163);  permute_1364 = None
    bmm_281: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1163, permute_1379);  view_1163 = permute_1379 = None
    view_1164: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_280, [1024, 1, 16, 64, 1]);  bmm_280 = None
    permute_1380: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1164, [4, 1, 2, 3, 0]);  view_1164 = None
    view_1165: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_281, [512, 1024, 1, 1, 1]);  bmm_281 = None
    permute_1381: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1165, [0, 2, 3, 4, 1]);  view_1165 = None
    permute_1382: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1380, [4, 2, 3, 0, 1]);  permute_1380 = None
    squeeze_135: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1382, 4);  permute_1382 = None
    squeeze_136: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_135, 3);  squeeze_135 = None
    permute_1383: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1381, [0, 1, 4, 2, 3]);  permute_1381 = None
    squeeze_137: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1383, 4);  permute_1383 = None
    squeeze_138: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_137, 3);  squeeze_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_308: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_307, squeeze_138);  add_307 = squeeze_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_393: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_308, primals_312);  primals_312 = None
    mul_394: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_393, 1024)
    sum_108: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_393, [2], True)
    mul_395: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_393, mul_146);  mul_393 = None
    sum_109: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True);  mul_395 = None
    mul_396: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_146, sum_109);  sum_109 = None
    sub_118: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_394, sum_108);  mul_394 = sum_108 = None
    sub_119: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_118, mul_396);  sub_118 = mul_396 = None
    mul_397: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_39, sub_119);  div_39 = sub_119 = None
    mul_398: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_308, mul_146);  mul_146 = None
    sum_110: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_398, [0, 1]);  mul_398 = None
    sum_111: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_308, [0, 1]);  add_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_30: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_217, torch.float32);  getitem_217 = None
    mul_399: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
    mul_400: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_397, mul_399);  mul_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1166: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_400, [512, 1024]);  mul_400 = None
    mm_26: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1166, permute_1384);  permute_1384 = None
    permute_1385: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1166, [1, 0])
    mm_27: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1385, view_682);  permute_1385 = view_682 = None
    permute_1386: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_112: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1166, [0], True);  view_1166 = None
    view_1167: "f32[1024]" = torch.ops.aten.reshape.default(sum_112, [1024]);  sum_112 = None
    permute_1387: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1386, [1, 0]);  permute_1386 = None
    view_1168: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_26, [512, 1, 4096]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_31: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_215, torch.float32);  getitem_215 = None
    mul_401: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
    mul_402: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1168, mul_401);  view_1168 = mul_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_404: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_196, 0.5);  add_196 = None
    mul_405: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_681, view_681)
    mul_406: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_405, -0.5);  mul_405 = None
    exp_32: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_406);  mul_406 = None
    mul_407: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_408: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_681, mul_407);  view_681 = mul_407 = None
    add_310: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_404, mul_408);  mul_404 = mul_408 = None
    mul_409: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_402, add_310);  mul_402 = add_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1169: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_409, [512, 4096]);  mul_409 = None
    mm_28: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1169, permute_1388);  permute_1388 = None
    permute_1389: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1169, [1, 0])
    mm_29: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1389, view_680);  permute_1389 = view_680 = None
    permute_1390: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_113: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1169, [0], True);  view_1169 = None
    view_1170: "f32[4096]" = torch.ops.aten.reshape.default(sum_113, [4096]);  sum_113 = None
    permute_1391: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1390, [1, 0]);  permute_1390 = None
    view_1171: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_28, [512, 1, 1024]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_311: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_397, view_1171);  mul_397 = view_1171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_411: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_311, primals_306);  primals_306 = None
    mul_412: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_411, 1024)
    sum_114: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True)
    mul_413: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_411, mul_141);  mul_411 = None
    sum_115: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
    mul_414: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_141, sum_115);  sum_115 = None
    sub_121: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_412, sum_114);  mul_412 = sum_114 = None
    sub_122: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_121, mul_414);  sub_121 = mul_414 = None
    mul_415: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_40, sub_122);  div_40 = sub_122 = None
    mul_416: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_311, mul_141);  mul_141 = None
    sum_116: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1]);  mul_416 = None
    sum_117: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_311, [0, 1]);  add_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_32: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_211, torch.float32);  getitem_211 = None
    mul_417: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_418: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_415, mul_417);  mul_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1172: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_418, [512, 1, 1024, 1, 1]);  mul_418 = None
    permute_1392: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1172, [0, 3, 4, 1, 2]);  view_1172 = None
    view_1173: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1392, [1, 512, 1024]);  permute_1392 = None
    bmm_282: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1393, view_1173);  permute_1393 = None
    bmm_283: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1173, permute_1394);  view_1173 = permute_1394 = None
    view_1174: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_282, [64, 16, 1, 1024, 1]);  bmm_282 = None
    permute_1395: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1174, [4, 2, 3, 0, 1]);  view_1174 = None
    view_1175: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_283, [512, 64, 16, 1, 1]);  bmm_283 = None
    permute_1396: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1175, [0, 3, 4, 1, 2]);  view_1175 = None
    permute_1397: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1395, [2, 4, 3, 0, 1]);  permute_1395 = None
    squeeze_139: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1397, 4);  permute_1397 = None
    squeeze_140: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_139, 3);  squeeze_139 = None
    permute_1398: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1396, [0, 1, 4, 3, 2]);  permute_1396 = None
    squeeze_141: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1398, 4);  permute_1398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1176: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_141, [512, 1, 16, 64, 1]);  squeeze_141 = None
    permute_1399: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1176, [2, 0, 4, 1, 3]);  view_1176 = None
    view_1177: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_1399, [16, 512, 64]);  permute_1399 = None
    bmm_284: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_1400, view_1177);  permute_1400 = None
    bmm_285: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1177, permute_1401);  view_1177 = permute_1401 = None
    view_1178: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_284, [16, 512, 1, 64, 1]);  bmm_284 = None
    permute_1402: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1178, [4, 2, 0, 3, 1]);  view_1178 = None
    view_1179: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_285, [16, 512, 512, 1, 1]);  bmm_285 = None
    permute_1403: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1179, [1, 3, 0, 4, 2]);  view_1179 = None
    permute_1404: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1402, [4, 1, 2, 3, 0]);  permute_1402 = None
    squeeze_142: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1404, 4);  permute_1404 = None
    permute_1405: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_1403, [1, 2, 0, 4, 3]);  permute_1403 = None
    squeeze_143: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_1405, 4);  permute_1405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_33: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_209, torch.float32);  getitem_209 = None
    mul_419: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
    mul_420: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_143, mul_419);  squeeze_143 = mul_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_421: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_420, alias_32);  mul_420 = None
    sum_118: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_421, [3], True)
    mul_422: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_32, sum_118);  alias_32 = sum_118 = None
    sub_123: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_423: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_123, 0.125);  sub_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_6: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_423, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1180: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_6, [1, 16, 1023, 512]);  index_put_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_25: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1180, 2, 1, 9223372036854775807);  view_1180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1181: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_25, [1, 16, 512, 1024]);  slice_scatter_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1182: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1181, [1, 16, 512, 1024, 1]);  view_1181 = None
    permute_1406: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1182, [1, 2, 4, 0, 3]);  view_1182 = None
    view_1183: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_1406, [16, 512, 1024]);  permute_1406 = None
    bmm_286: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_1407, view_1183);  permute_1407 = None
    bmm_287: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1183, permute_1408);  view_1183 = permute_1408 = None
    view_1184: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_286, [16, 64, 1, 1024, 1]);  bmm_286 = None
    permute_1409: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1184, [2, 0, 4, 3, 1]);  view_1184 = None
    view_1185: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_287, [16, 512, 64, 1, 1]);  bmm_287 = None
    permute_1410: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1185, [3, 0, 1, 4, 2]);  view_1185 = None
    permute_1411: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1409, [3, 0, 1, 4, 2]);  permute_1409 = None
    squeeze_144: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1411, 4);  permute_1411 = None
    permute_1412: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1410, [2, 0, 1, 4, 3]);  permute_1410 = None
    squeeze_145: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1412, 4);  permute_1412 = None
    sum_119: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_145, [0, 1], True)
    view_1186: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_119, [16, 64]);  sum_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1187: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_423, [1, 16, 512, 512, 1]);  mul_423 = None
    permute_1413: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1187, [1, 2, 4, 0, 3]);  view_1187 = None
    view_1188: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_1413, [16, 512, 512]);  permute_1413 = None
    bmm_288: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_1414, view_1188);  permute_1414 = None
    bmm_289: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1188, permute_1415);  view_1188 = permute_1415 = None
    view_1189: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_288, [16, 64, 1, 512, 1]);  bmm_288 = None
    permute_1416: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1189, [2, 0, 4, 3, 1]);  view_1189 = None
    view_1190: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_289, [16, 512, 64, 1, 1]);  bmm_289 = None
    permute_1417: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1190, [3, 0, 1, 4, 2]);  view_1190 = None
    permute_1418: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1416, [3, 0, 1, 4, 2]);  permute_1416 = None
    squeeze_146: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1418, 4);  permute_1418 = None
    permute_1419: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1417, [2, 0, 1, 4, 3]);  permute_1417 = None
    squeeze_147: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1419, 4);  permute_1419 = None
    sum_120: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_147, [0, 1], True)
    view_1191: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_120, [16, 64]);  sum_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_312: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_145, squeeze_147);  squeeze_145 = squeeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1192: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_144, [1024, 1, 16, 64, 1]);  squeeze_144 = None
    permute_1420: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1192, [0, 4, 1, 2, 3]);  view_1192 = None
    view_1193: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_1420, [1, 1024, 1024]);  permute_1420 = None
    bmm_290: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1193);  view_1193 = None
    view_1194: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_290, [1024, 1, 16, 64, 1]);  bmm_290 = None
    permute_1422: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1194, [4, 1, 2, 3, 0]);  view_1194 = None
    permute_1423: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1422, [4, 2, 3, 0, 1]);  permute_1422 = None
    squeeze_148: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1423, 4);  permute_1423 = None
    squeeze_149: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_148, 3);  squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1195: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_142, [512, 1, 16, 64, 1]);  squeeze_142 = None
    permute_1424: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1195, [0, 4, 1, 2, 3]);  view_1195 = None
    clone_89: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1424, memory_format = torch.contiguous_format);  permute_1424 = None
    view_1196: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_89, [1, 512, 1024]);  clone_89 = None
    bmm_291: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1425, view_1196)
    bmm_292: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1196, permute_1426);  view_1196 = permute_1426 = None
    view_1197: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_291, [1024, 1, 16, 64, 1]);  bmm_291 = None
    permute_1427: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1197, [4, 1, 2, 3, 0]);  view_1197 = None
    view_1198: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_292, [512, 1024, 1, 1, 1]);  bmm_292 = None
    permute_1428: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1198, [0, 2, 3, 4, 1]);  view_1198 = None
    permute_1429: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1427, [4, 2, 3, 0, 1]);  permute_1427 = None
    squeeze_150: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1429, 4);  permute_1429 = None
    squeeze_151: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_150, 3);  squeeze_150 = None
    permute_1430: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1428, [0, 1, 4, 2, 3]);  permute_1428 = None
    squeeze_152: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1430, 4);  permute_1430 = None
    squeeze_153: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_152, 3);  squeeze_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_313: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_415, squeeze_153);  mul_415 = squeeze_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1199: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_146, [512, 1, 16, 64, 1]);  squeeze_146 = None
    permute_1431: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1199, [0, 4, 1, 2, 3]);  view_1199 = None
    view_1200: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1431, [1, 512, 1024]);  permute_1431 = None
    bmm_293: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1425, view_1200)
    bmm_294: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1200, permute_1433);  view_1200 = permute_1433 = None
    view_1201: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_293, [1024, 1, 16, 64, 1]);  bmm_293 = None
    permute_1434: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1201, [4, 1, 2, 3, 0]);  view_1201 = None
    view_1202: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_294, [512, 1024, 1, 1, 1]);  bmm_294 = None
    permute_1435: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1202, [0, 2, 3, 4, 1]);  view_1202 = None
    permute_1436: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1434, [4, 2, 3, 0, 1]);  permute_1434 = None
    squeeze_154: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1436, 4);  permute_1436 = None
    squeeze_155: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_154, 3);  squeeze_154 = None
    permute_1437: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1435, [0, 1, 4, 2, 3]);  permute_1435 = None
    squeeze_156: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1437, 4);  permute_1437 = None
    squeeze_157: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_156, 3);  squeeze_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_314: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_313, squeeze_157);  add_313 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1203: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_312, [512, 1, 16, 64, 1]);  add_312 = None
    permute_1438: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1203, [0, 4, 1, 2, 3]);  view_1203 = None
    clone_90: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1438, memory_format = torch.contiguous_format);  permute_1438 = None
    view_1204: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_90, [1, 512, 1024]);  clone_90 = None
    bmm_295: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1425, view_1204);  permute_1425 = None
    bmm_296: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1204, permute_1440);  view_1204 = permute_1440 = None
    view_1205: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_295, [1024, 1, 16, 64, 1]);  bmm_295 = None
    permute_1441: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1205, [4, 1, 2, 3, 0]);  view_1205 = None
    view_1206: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_296, [512, 1024, 1, 1, 1]);  bmm_296 = None
    permute_1442: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1206, [0, 2, 3, 4, 1]);  view_1206 = None
    permute_1443: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1441, [4, 2, 3, 0, 1]);  permute_1441 = None
    squeeze_158: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1443, 4);  permute_1443 = None
    squeeze_159: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_158, 3);  squeeze_158 = None
    permute_1444: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1442, [0, 1, 4, 2, 3]);  permute_1442 = None
    squeeze_160: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1444, 4);  permute_1444 = None
    squeeze_161: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_160, 3);  squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_315: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_314, squeeze_161);  add_314 = squeeze_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_425: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_315, primals_304);  primals_304 = None
    mul_426: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_425, 1024)
    sum_121: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True)
    mul_427: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_425, mul_138);  mul_425 = None
    sum_122: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True);  mul_427 = None
    mul_428: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_138, sum_122);  sum_122 = None
    sub_125: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_426, sum_121);  mul_426 = sum_121 = None
    sub_126: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_125, mul_428);  sub_125 = mul_428 = None
    mul_429: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_41, sub_126);  div_41 = sub_126 = None
    mul_430: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_315, mul_138);  mul_138 = None
    sum_123: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 1]);  mul_430 = None
    sum_124: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_315, [0, 1]);  add_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_34: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_205, torch.float32);  getitem_205 = None
    mul_431: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
    mul_432: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_429, mul_431);  mul_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1207: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_432, [512, 1024]);  mul_432 = None
    mm_30: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1207, permute_1445);  permute_1445 = None
    permute_1446: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1207, [1, 0])
    mm_31: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1446, view_644);  permute_1446 = view_644 = None
    permute_1447: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_125: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1207, [0], True);  view_1207 = None
    view_1208: "f32[1024]" = torch.ops.aten.reshape.default(sum_125, [1024]);  sum_125 = None
    permute_1448: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1447, [1, 0]);  permute_1447 = None
    view_1209: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_30, [512, 1, 4096]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_35: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_203, torch.float32);  getitem_203 = None
    mul_433: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_434: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1209, mul_433);  view_1209 = mul_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_436: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_185, 0.5);  add_185 = None
    mul_437: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_643, view_643)
    mul_438: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_437, -0.5);  mul_437 = None
    exp_33: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_438);  mul_438 = None
    mul_439: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_440: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_643, mul_439);  view_643 = mul_439 = None
    add_317: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_436, mul_440);  mul_436 = mul_440 = None
    mul_441: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_434, add_317);  mul_434 = add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1210: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_441, [512, 4096]);  mul_441 = None
    mm_32: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1210, permute_1449);  permute_1449 = None
    permute_1450: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1210, [1, 0])
    mm_33: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1450, view_642);  permute_1450 = view_642 = None
    permute_1451: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_126: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1210, [0], True);  view_1210 = None
    view_1211: "f32[4096]" = torch.ops.aten.reshape.default(sum_126, [4096]);  sum_126 = None
    permute_1452: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1451, [1, 0]);  permute_1451 = None
    view_1212: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_32, [512, 1, 1024]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_318: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_429, view_1212);  mul_429 = view_1212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_443: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_318, primals_298);  primals_298 = None
    mul_444: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_443, 1024)
    sum_127: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_443, [2], True)
    mul_445: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_443, mul_133);  mul_443 = None
    sum_128: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_445, [2], True);  mul_445 = None
    mul_446: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_133, sum_128);  sum_128 = None
    sub_128: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_444, sum_127);  mul_444 = sum_127 = None
    sub_129: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_128, mul_446);  sub_128 = mul_446 = None
    mul_447: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_42, sub_129);  div_42 = sub_129 = None
    mul_448: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_318, mul_133);  mul_133 = None
    sum_129: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_448, [0, 1]);  mul_448 = None
    sum_130: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_318, [0, 1]);  add_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_36: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_199, torch.float32);  getitem_199 = None
    mul_449: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
    mul_450: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_447, mul_449);  mul_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1213: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_450, [512, 1, 1024, 1, 1]);  mul_450 = None
    permute_1453: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1213, [0, 3, 4, 1, 2]);  view_1213 = None
    view_1214: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1453, [1, 512, 1024]);  permute_1453 = None
    bmm_297: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1454, view_1214);  permute_1454 = None
    bmm_298: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1214, permute_1455);  view_1214 = permute_1455 = None
    view_1215: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_297, [64, 16, 1, 1024, 1]);  bmm_297 = None
    permute_1456: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1215, [4, 2, 3, 0, 1]);  view_1215 = None
    view_1216: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_298, [512, 64, 16, 1, 1]);  bmm_298 = None
    permute_1457: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1216, [0, 3, 4, 1, 2]);  view_1216 = None
    permute_1458: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1456, [2, 4, 3, 0, 1]);  permute_1456 = None
    squeeze_162: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1458, 4);  permute_1458 = None
    squeeze_163: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_162, 3);  squeeze_162 = None
    permute_1459: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1457, [0, 1, 4, 3, 2]);  permute_1457 = None
    squeeze_164: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1459, 4);  permute_1459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1217: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_164, [512, 1, 16, 64, 1]);  squeeze_164 = None
    permute_1460: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1217, [2, 0, 4, 1, 3]);  view_1217 = None
    view_1218: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_1460, [16, 512, 64]);  permute_1460 = None
    bmm_299: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_1461, view_1218);  permute_1461 = None
    bmm_300: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1218, permute_1462);  view_1218 = permute_1462 = None
    view_1219: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_299, [16, 512, 1, 64, 1]);  bmm_299 = None
    permute_1463: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1219, [4, 2, 0, 3, 1]);  view_1219 = None
    view_1220: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_300, [16, 512, 512, 1, 1]);  bmm_300 = None
    permute_1464: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1220, [1, 3, 0, 4, 2]);  view_1220 = None
    permute_1465: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1463, [4, 1, 2, 3, 0]);  permute_1463 = None
    squeeze_165: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1465, 4);  permute_1465 = None
    permute_1466: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_1464, [1, 2, 0, 4, 3]);  permute_1464 = None
    squeeze_166: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_1466, 4);  permute_1466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_37: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_197, torch.float32);  getitem_197 = None
    mul_451: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
    mul_452: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_166, mul_451);  squeeze_166 = mul_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_453: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_452, alias_33);  mul_452 = None
    sum_131: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [3], True)
    mul_454: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_33, sum_131);  alias_33 = sum_131 = None
    sub_130: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_453, mul_454);  mul_453 = mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_455: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_130, 0.125);  sub_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_7: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_455, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1221: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_7, [1, 16, 1023, 512]);  index_put_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_29: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1221, 2, 1, 9223372036854775807);  view_1221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1222: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_29, [1, 16, 512, 1024]);  slice_scatter_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1223: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1222, [1, 16, 512, 1024, 1]);  view_1222 = None
    permute_1467: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1223, [1, 2, 4, 0, 3]);  view_1223 = None
    view_1224: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_1467, [16, 512, 1024]);  permute_1467 = None
    bmm_301: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_1468, view_1224);  permute_1468 = None
    bmm_302: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1224, permute_1469);  view_1224 = permute_1469 = None
    view_1225: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_301, [16, 64, 1, 1024, 1]);  bmm_301 = None
    permute_1470: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1225, [2, 0, 4, 3, 1]);  view_1225 = None
    view_1226: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_302, [16, 512, 64, 1, 1]);  bmm_302 = None
    permute_1471: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1226, [3, 0, 1, 4, 2]);  view_1226 = None
    permute_1472: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1470, [3, 0, 1, 4, 2]);  permute_1470 = None
    squeeze_167: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1472, 4);  permute_1472 = None
    permute_1473: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1471, [2, 0, 1, 4, 3]);  permute_1471 = None
    squeeze_168: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1473, 4);  permute_1473 = None
    sum_132: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_168, [0, 1], True)
    view_1227: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_132, [16, 64]);  sum_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1228: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_455, [1, 16, 512, 512, 1]);  mul_455 = None
    permute_1474: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1228, [1, 2, 4, 0, 3]);  view_1228 = None
    view_1229: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_1474, [16, 512, 512]);  permute_1474 = None
    bmm_303: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_1475, view_1229);  permute_1475 = None
    bmm_304: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1229, permute_1476);  view_1229 = permute_1476 = None
    view_1230: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_303, [16, 64, 1, 512, 1]);  bmm_303 = None
    permute_1477: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1230, [2, 0, 4, 3, 1]);  view_1230 = None
    view_1231: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_304, [16, 512, 64, 1, 1]);  bmm_304 = None
    permute_1478: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1231, [3, 0, 1, 4, 2]);  view_1231 = None
    permute_1479: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1477, [3, 0, 1, 4, 2]);  permute_1477 = None
    squeeze_169: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1479, 4);  permute_1479 = None
    permute_1480: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1478, [2, 0, 1, 4, 3]);  permute_1478 = None
    squeeze_170: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1480, 4);  permute_1480 = None
    sum_133: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_170, [0, 1], True)
    view_1232: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_133, [16, 64]);  sum_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_319: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_168, squeeze_170);  squeeze_168 = squeeze_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1233: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_167, [1024, 1, 16, 64, 1]);  squeeze_167 = None
    permute_1481: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1233, [0, 4, 1, 2, 3]);  view_1233 = None
    view_1234: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_1481, [1, 1024, 1024]);  permute_1481 = None
    bmm_305: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1234);  view_1234 = None
    view_1235: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_305, [1024, 1, 16, 64, 1]);  bmm_305 = None
    permute_1483: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1235, [4, 1, 2, 3, 0]);  view_1235 = None
    permute_1484: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1483, [4, 2, 3, 0, 1]);  permute_1483 = None
    squeeze_171: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1484, 4);  permute_1484 = None
    squeeze_172: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_171, 3);  squeeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1236: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_165, [512, 1, 16, 64, 1]);  squeeze_165 = None
    permute_1485: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1236, [0, 4, 1, 2, 3]);  view_1236 = None
    clone_95: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1485, memory_format = torch.contiguous_format);  permute_1485 = None
    view_1237: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_95, [1, 512, 1024]);  clone_95 = None
    bmm_306: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1486, view_1237)
    bmm_307: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1237, permute_1487);  view_1237 = permute_1487 = None
    view_1238: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_306, [1024, 1, 16, 64, 1]);  bmm_306 = None
    permute_1488: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1238, [4, 1, 2, 3, 0]);  view_1238 = None
    view_1239: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_307, [512, 1024, 1, 1, 1]);  bmm_307 = None
    permute_1489: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1239, [0, 2, 3, 4, 1]);  view_1239 = None
    permute_1490: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1488, [4, 2, 3, 0, 1]);  permute_1488 = None
    squeeze_173: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1490, 4);  permute_1490 = None
    squeeze_174: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_173, 3);  squeeze_173 = None
    permute_1491: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1489, [0, 1, 4, 2, 3]);  permute_1489 = None
    squeeze_175: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1491, 4);  permute_1491 = None
    squeeze_176: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_175, 3);  squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_320: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_447, squeeze_176);  mul_447 = squeeze_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1240: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_169, [512, 1, 16, 64, 1]);  squeeze_169 = None
    permute_1492: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1240, [0, 4, 1, 2, 3]);  view_1240 = None
    view_1241: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1492, [1, 512, 1024]);  permute_1492 = None
    bmm_308: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1486, view_1241)
    bmm_309: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1241, permute_1494);  view_1241 = permute_1494 = None
    view_1242: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_308, [1024, 1, 16, 64, 1]);  bmm_308 = None
    permute_1495: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1242, [4, 1, 2, 3, 0]);  view_1242 = None
    view_1243: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_309, [512, 1024, 1, 1, 1]);  bmm_309 = None
    permute_1496: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1243, [0, 2, 3, 4, 1]);  view_1243 = None
    permute_1497: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1495, [4, 2, 3, 0, 1]);  permute_1495 = None
    squeeze_177: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1497, 4);  permute_1497 = None
    squeeze_178: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_177, 3);  squeeze_177 = None
    permute_1498: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1496, [0, 1, 4, 2, 3]);  permute_1496 = None
    squeeze_179: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1498, 4);  permute_1498 = None
    squeeze_180: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_179, 3);  squeeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_321: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_320, squeeze_180);  add_320 = squeeze_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1244: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_319, [512, 1, 16, 64, 1]);  add_319 = None
    permute_1499: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1244, [0, 4, 1, 2, 3]);  view_1244 = None
    clone_96: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1499, memory_format = torch.contiguous_format);  permute_1499 = None
    view_1245: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_96, [1, 512, 1024]);  clone_96 = None
    bmm_310: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1486, view_1245);  permute_1486 = None
    bmm_311: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1245, permute_1501);  view_1245 = permute_1501 = None
    view_1246: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_310, [1024, 1, 16, 64, 1]);  bmm_310 = None
    permute_1502: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1246, [4, 1, 2, 3, 0]);  view_1246 = None
    view_1247: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_311, [512, 1024, 1, 1, 1]);  bmm_311 = None
    permute_1503: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1247, [0, 2, 3, 4, 1]);  view_1247 = None
    permute_1504: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1502, [4, 2, 3, 0, 1]);  permute_1502 = None
    squeeze_181: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1504, 4);  permute_1504 = None
    squeeze_182: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_181, 3);  squeeze_181 = None
    permute_1505: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1503, [0, 1, 4, 2, 3]);  permute_1503 = None
    squeeze_183: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1505, 4);  permute_1505 = None
    squeeze_184: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_183, 3);  squeeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_322: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_321, squeeze_184);  add_321 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_457: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_322, primals_296);  primals_296 = None
    mul_458: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_457, 1024)
    sum_134: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_457, [2], True)
    mul_459: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_457, mul_130);  mul_457 = None
    sum_135: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_459, [2], True);  mul_459 = None
    mul_460: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_130, sum_135);  sum_135 = None
    sub_132: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_458, sum_134);  mul_458 = sum_134 = None
    sub_133: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_132, mul_460);  sub_132 = mul_460 = None
    mul_461: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_43, sub_133);  div_43 = sub_133 = None
    mul_462: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_322, mul_130);  mul_130 = None
    sum_136: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_462, [0, 1]);  mul_462 = None
    sum_137: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_322, [0, 1]);  add_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_38: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_193, torch.float32);  getitem_193 = None
    mul_463: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
    mul_464: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_461, mul_463);  mul_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1248: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_464, [512, 1024]);  mul_464 = None
    mm_34: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1248, permute_1506);  permute_1506 = None
    permute_1507: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1248, [1, 0])
    mm_35: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1507, view_606);  permute_1507 = view_606 = None
    permute_1508: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_138: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1248, [0], True);  view_1248 = None
    view_1249: "f32[1024]" = torch.ops.aten.reshape.default(sum_138, [1024]);  sum_138 = None
    permute_1509: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1508, [1, 0]);  permute_1508 = None
    view_1250: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_34, [512, 1, 4096]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_39: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_191, torch.float32);  getitem_191 = None
    mul_465: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 1.1111111111111112);  convert_element_type_39 = None
    mul_466: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1250, mul_465);  view_1250 = mul_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_468: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_174, 0.5);  add_174 = None
    mul_469: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_605, view_605)
    mul_470: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_469, -0.5);  mul_469 = None
    exp_34: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_470);  mul_470 = None
    mul_471: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_472: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_605, mul_471);  view_605 = mul_471 = None
    add_324: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_468, mul_472);  mul_468 = mul_472 = None
    mul_473: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_466, add_324);  mul_466 = add_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1251: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_473, [512, 4096]);  mul_473 = None
    mm_36: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1251, permute_1510);  permute_1510 = None
    permute_1511: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1251, [1, 0])
    mm_37: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1511, view_604);  permute_1511 = view_604 = None
    permute_1512: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_139: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1251, [0], True);  view_1251 = None
    view_1252: "f32[4096]" = torch.ops.aten.reshape.default(sum_139, [4096]);  sum_139 = None
    permute_1513: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1512, [1, 0]);  permute_1512 = None
    view_1253: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_36, [512, 1, 1024]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_325: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_461, view_1253);  mul_461 = view_1253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_475: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_325, primals_290);  primals_290 = None
    mul_476: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_475, 1024)
    sum_140: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_475, [2], True)
    mul_477: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_475, mul_125);  mul_475 = None
    sum_141: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_477, [2], True);  mul_477 = None
    mul_478: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_125, sum_141);  sum_141 = None
    sub_135: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_476, sum_140);  mul_476 = sum_140 = None
    sub_136: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_135, mul_478);  sub_135 = mul_478 = None
    mul_479: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_44, sub_136);  div_44 = sub_136 = None
    mul_480: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_325, mul_125);  mul_125 = None
    sum_142: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_480, [0, 1]);  mul_480 = None
    sum_143: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_325, [0, 1]);  add_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_40: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_187, torch.float32);  getitem_187 = None
    mul_481: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
    mul_482: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_479, mul_481);  mul_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1254: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_482, [512, 1, 1024, 1, 1]);  mul_482 = None
    permute_1514: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1254, [0, 3, 4, 1, 2]);  view_1254 = None
    view_1255: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1514, [1, 512, 1024]);  permute_1514 = None
    bmm_312: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1515, view_1255);  permute_1515 = None
    bmm_313: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1255, permute_1516);  view_1255 = permute_1516 = None
    view_1256: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_312, [64, 16, 1, 1024, 1]);  bmm_312 = None
    permute_1517: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1256, [4, 2, 3, 0, 1]);  view_1256 = None
    view_1257: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_313, [512, 64, 16, 1, 1]);  bmm_313 = None
    permute_1518: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1257, [0, 3, 4, 1, 2]);  view_1257 = None
    permute_1519: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1517, [2, 4, 3, 0, 1]);  permute_1517 = None
    squeeze_185: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1519, 4);  permute_1519 = None
    squeeze_186: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_185, 3);  squeeze_185 = None
    permute_1520: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1518, [0, 1, 4, 3, 2]);  permute_1518 = None
    squeeze_187: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1520, 4);  permute_1520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1258: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_187, [512, 1, 16, 64, 1]);  squeeze_187 = None
    permute_1521: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1258, [2, 0, 4, 1, 3]);  view_1258 = None
    view_1259: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_1521, [16, 512, 64]);  permute_1521 = None
    bmm_314: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_1522, view_1259);  permute_1522 = None
    bmm_315: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1259, permute_1523);  view_1259 = permute_1523 = None
    view_1260: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_314, [16, 512, 1, 64, 1]);  bmm_314 = None
    permute_1524: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1260, [4, 2, 0, 3, 1]);  view_1260 = None
    view_1261: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_315, [16, 512, 512, 1, 1]);  bmm_315 = None
    permute_1525: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1261, [1, 3, 0, 4, 2]);  view_1261 = None
    permute_1526: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1524, [4, 1, 2, 3, 0]);  permute_1524 = None
    squeeze_188: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1526, 4);  permute_1526 = None
    permute_1527: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_1525, [1, 2, 0, 4, 3]);  permute_1525 = None
    squeeze_189: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_1527, 4);  permute_1527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_41: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_185, torch.float32);  getitem_185 = None
    mul_483: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.1111111111111112);  convert_element_type_41 = None
    mul_484: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_189, mul_483);  squeeze_189 = mul_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_485: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_484, alias_34);  mul_484 = None
    sum_144: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [3], True)
    mul_486: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_34, sum_144);  alias_34 = sum_144 = None
    sub_137: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_485, mul_486);  mul_485 = mul_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_487: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_137, 0.125);  sub_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_8: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_487, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1262: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_8, [1, 16, 1023, 512]);  index_put_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_33: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1262, 2, 1, 9223372036854775807);  view_1262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1263: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_33, [1, 16, 512, 1024]);  slice_scatter_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1264: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1263, [1, 16, 512, 1024, 1]);  view_1263 = None
    permute_1528: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1264, [1, 2, 4, 0, 3]);  view_1264 = None
    view_1265: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_1528, [16, 512, 1024]);  permute_1528 = None
    bmm_316: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_1529, view_1265);  permute_1529 = None
    bmm_317: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1265, permute_1530);  view_1265 = permute_1530 = None
    view_1266: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_316, [16, 64, 1, 1024, 1]);  bmm_316 = None
    permute_1531: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1266, [2, 0, 4, 3, 1]);  view_1266 = None
    view_1267: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_317, [16, 512, 64, 1, 1]);  bmm_317 = None
    permute_1532: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1267, [3, 0, 1, 4, 2]);  view_1267 = None
    permute_1533: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1531, [3, 0, 1, 4, 2]);  permute_1531 = None
    squeeze_190: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1533, 4);  permute_1533 = None
    permute_1534: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1532, [2, 0, 1, 4, 3]);  permute_1532 = None
    squeeze_191: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1534, 4);  permute_1534 = None
    sum_145: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_191, [0, 1], True)
    view_1268: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_145, [16, 64]);  sum_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1269: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_487, [1, 16, 512, 512, 1]);  mul_487 = None
    permute_1535: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1269, [1, 2, 4, 0, 3]);  view_1269 = None
    view_1270: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_1535, [16, 512, 512]);  permute_1535 = None
    bmm_318: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_1536, view_1270);  permute_1536 = None
    bmm_319: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1270, permute_1537);  view_1270 = permute_1537 = None
    view_1271: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_318, [16, 64, 1, 512, 1]);  bmm_318 = None
    permute_1538: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1271, [2, 0, 4, 3, 1]);  view_1271 = None
    view_1272: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_319, [16, 512, 64, 1, 1]);  bmm_319 = None
    permute_1539: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1272, [3, 0, 1, 4, 2]);  view_1272 = None
    permute_1540: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1538, [3, 0, 1, 4, 2]);  permute_1538 = None
    squeeze_192: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1540, 4);  permute_1540 = None
    permute_1541: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1539, [2, 0, 1, 4, 3]);  permute_1539 = None
    squeeze_193: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1541, 4);  permute_1541 = None
    sum_146: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_193, [0, 1], True)
    view_1273: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_146, [16, 64]);  sum_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_326: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_191, squeeze_193);  squeeze_191 = squeeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1274: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_190, [1024, 1, 16, 64, 1]);  squeeze_190 = None
    permute_1542: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1274, [0, 4, 1, 2, 3]);  view_1274 = None
    view_1275: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_1542, [1, 1024, 1024]);  permute_1542 = None
    bmm_320: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1275);  view_1275 = None
    view_1276: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_320, [1024, 1, 16, 64, 1]);  bmm_320 = None
    permute_1544: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1276, [4, 1, 2, 3, 0]);  view_1276 = None
    permute_1545: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1544, [4, 2, 3, 0, 1]);  permute_1544 = None
    squeeze_194: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1545, 4);  permute_1545 = None
    squeeze_195: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_194, 3);  squeeze_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1277: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_188, [512, 1, 16, 64, 1]);  squeeze_188 = None
    permute_1546: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1277, [0, 4, 1, 2, 3]);  view_1277 = None
    clone_101: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1546, memory_format = torch.contiguous_format);  permute_1546 = None
    view_1278: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_101, [1, 512, 1024]);  clone_101 = None
    bmm_321: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1547, view_1278)
    bmm_322: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1278, permute_1548);  view_1278 = permute_1548 = None
    view_1279: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_321, [1024, 1, 16, 64, 1]);  bmm_321 = None
    permute_1549: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1279, [4, 1, 2, 3, 0]);  view_1279 = None
    view_1280: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_322, [512, 1024, 1, 1, 1]);  bmm_322 = None
    permute_1550: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1280, [0, 2, 3, 4, 1]);  view_1280 = None
    permute_1551: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1549, [4, 2, 3, 0, 1]);  permute_1549 = None
    squeeze_196: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1551, 4);  permute_1551 = None
    squeeze_197: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_196, 3);  squeeze_196 = None
    permute_1552: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1550, [0, 1, 4, 2, 3]);  permute_1550 = None
    squeeze_198: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1552, 4);  permute_1552 = None
    squeeze_199: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_198, 3);  squeeze_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_327: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_479, squeeze_199);  mul_479 = squeeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1281: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_192, [512, 1, 16, 64, 1]);  squeeze_192 = None
    permute_1553: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1281, [0, 4, 1, 2, 3]);  view_1281 = None
    view_1282: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1553, [1, 512, 1024]);  permute_1553 = None
    bmm_323: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1547, view_1282)
    bmm_324: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1282, permute_1555);  view_1282 = permute_1555 = None
    view_1283: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_323, [1024, 1, 16, 64, 1]);  bmm_323 = None
    permute_1556: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1283, [4, 1, 2, 3, 0]);  view_1283 = None
    view_1284: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_324, [512, 1024, 1, 1, 1]);  bmm_324 = None
    permute_1557: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1284, [0, 2, 3, 4, 1]);  view_1284 = None
    permute_1558: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1556, [4, 2, 3, 0, 1]);  permute_1556 = None
    squeeze_200: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1558, 4);  permute_1558 = None
    squeeze_201: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_200, 3);  squeeze_200 = None
    permute_1559: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1557, [0, 1, 4, 2, 3]);  permute_1557 = None
    squeeze_202: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1559, 4);  permute_1559 = None
    squeeze_203: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_202, 3);  squeeze_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_328: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_327, squeeze_203);  add_327 = squeeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1285: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_326, [512, 1, 16, 64, 1]);  add_326 = None
    permute_1560: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1285, [0, 4, 1, 2, 3]);  view_1285 = None
    clone_102: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1560, memory_format = torch.contiguous_format);  permute_1560 = None
    view_1286: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_102, [1, 512, 1024]);  clone_102 = None
    bmm_325: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1547, view_1286);  permute_1547 = None
    bmm_326: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1286, permute_1562);  view_1286 = permute_1562 = None
    view_1287: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_325, [1024, 1, 16, 64, 1]);  bmm_325 = None
    permute_1563: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1287, [4, 1, 2, 3, 0]);  view_1287 = None
    view_1288: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_326, [512, 1024, 1, 1, 1]);  bmm_326 = None
    permute_1564: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1288, [0, 2, 3, 4, 1]);  view_1288 = None
    permute_1565: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1563, [4, 2, 3, 0, 1]);  permute_1563 = None
    squeeze_204: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1565, 4);  permute_1565 = None
    squeeze_205: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_204, 3);  squeeze_204 = None
    permute_1566: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1564, [0, 1, 4, 2, 3]);  permute_1564 = None
    squeeze_206: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1566, 4);  permute_1566 = None
    squeeze_207: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_206, 3);  squeeze_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_329: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_328, squeeze_207);  add_328 = squeeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_489: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_329, primals_288);  primals_288 = None
    mul_490: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_489, 1024)
    sum_147: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_489, [2], True)
    mul_491: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_489, mul_122);  mul_489 = None
    sum_148: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_491, [2], True);  mul_491 = None
    mul_492: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_122, sum_148);  sum_148 = None
    sub_139: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_490, sum_147);  mul_490 = sum_147 = None
    sub_140: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_139, mul_492);  sub_139 = mul_492 = None
    mul_493: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_45, sub_140);  div_45 = sub_140 = None
    mul_494: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_329, mul_122);  mul_122 = None
    sum_149: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_494, [0, 1]);  mul_494 = None
    sum_150: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_329, [0, 1]);  add_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_42: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_181, torch.float32);  getitem_181 = None
    mul_495: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 1.1111111111111112);  convert_element_type_42 = None
    mul_496: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_493, mul_495);  mul_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1289: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_496, [512, 1024]);  mul_496 = None
    mm_38: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1289, permute_1567);  permute_1567 = None
    permute_1568: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1289, [1, 0])
    mm_39: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1568, view_568);  permute_1568 = view_568 = None
    permute_1569: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_151: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1289, [0], True);  view_1289 = None
    view_1290: "f32[1024]" = torch.ops.aten.reshape.default(sum_151, [1024]);  sum_151 = None
    permute_1570: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1569, [1, 0]);  permute_1569 = None
    view_1291: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_38, [512, 1, 4096]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_43: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_179, torch.float32);  getitem_179 = None
    mul_497: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1.1111111111111112);  convert_element_type_43 = None
    mul_498: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1291, mul_497);  view_1291 = mul_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_500: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_163, 0.5);  add_163 = None
    mul_501: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_567, view_567)
    mul_502: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_501, -0.5);  mul_501 = None
    exp_35: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_502);  mul_502 = None
    mul_503: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_504: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_567, mul_503);  view_567 = mul_503 = None
    add_331: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_500, mul_504);  mul_500 = mul_504 = None
    mul_505: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_498, add_331);  mul_498 = add_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1292: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_505, [512, 4096]);  mul_505 = None
    mm_40: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1292, permute_1571);  permute_1571 = None
    permute_1572: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1292, [1, 0])
    mm_41: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1572, view_566);  permute_1572 = view_566 = None
    permute_1573: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_152: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1292, [0], True);  view_1292 = None
    view_1293: "f32[4096]" = torch.ops.aten.reshape.default(sum_152, [4096]);  sum_152 = None
    permute_1574: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1573, [1, 0]);  permute_1573 = None
    view_1294: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_40, [512, 1, 1024]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_332: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_493, view_1294);  mul_493 = view_1294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_507: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_332, primals_282);  primals_282 = None
    mul_508: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_507, 1024)
    sum_153: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_507, [2], True)
    mul_509: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_507, mul_117);  mul_507 = None
    sum_154: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_509, [2], True);  mul_509 = None
    mul_510: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_117, sum_154);  sum_154 = None
    sub_142: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_508, sum_153);  mul_508 = sum_153 = None
    sub_143: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_142, mul_510);  sub_142 = mul_510 = None
    mul_511: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_46, sub_143);  div_46 = sub_143 = None
    mul_512: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_332, mul_117);  mul_117 = None
    sum_155: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_512, [0, 1]);  mul_512 = None
    sum_156: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_332, [0, 1]);  add_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_44: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_175, torch.float32);  getitem_175 = None
    mul_513: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 1.1111111111111112);  convert_element_type_44 = None
    mul_514: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_511, mul_513);  mul_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1295: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_514, [512, 1, 1024, 1, 1]);  mul_514 = None
    permute_1575: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1295, [0, 3, 4, 1, 2]);  view_1295 = None
    view_1296: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1575, [1, 512, 1024]);  permute_1575 = None
    bmm_327: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1576, view_1296);  permute_1576 = None
    bmm_328: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1296, permute_1577);  view_1296 = permute_1577 = None
    view_1297: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_327, [64, 16, 1, 1024, 1]);  bmm_327 = None
    permute_1578: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1297, [4, 2, 3, 0, 1]);  view_1297 = None
    view_1298: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_328, [512, 64, 16, 1, 1]);  bmm_328 = None
    permute_1579: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1298, [0, 3, 4, 1, 2]);  view_1298 = None
    permute_1580: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1578, [2, 4, 3, 0, 1]);  permute_1578 = None
    squeeze_208: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1580, 4);  permute_1580 = None
    squeeze_209: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_208, 3);  squeeze_208 = None
    permute_1581: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1579, [0, 1, 4, 3, 2]);  permute_1579 = None
    squeeze_210: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1581, 4);  permute_1581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1299: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_210, [512, 1, 16, 64, 1]);  squeeze_210 = None
    permute_1582: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1299, [2, 0, 4, 1, 3]);  view_1299 = None
    view_1300: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_1582, [16, 512, 64]);  permute_1582 = None
    bmm_329: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_1583, view_1300);  permute_1583 = None
    bmm_330: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1300, permute_1584);  view_1300 = permute_1584 = None
    view_1301: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_329, [16, 512, 1, 64, 1]);  bmm_329 = None
    permute_1585: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1301, [4, 2, 0, 3, 1]);  view_1301 = None
    view_1302: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_330, [16, 512, 512, 1, 1]);  bmm_330 = None
    permute_1586: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1302, [1, 3, 0, 4, 2]);  view_1302 = None
    permute_1587: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1585, [4, 1, 2, 3, 0]);  permute_1585 = None
    squeeze_211: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1587, 4);  permute_1587 = None
    permute_1588: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_1586, [1, 2, 0, 4, 3]);  permute_1586 = None
    squeeze_212: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_1588, 4);  permute_1588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_45: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_173, torch.float32);  getitem_173 = None
    mul_515: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_45, 1.1111111111111112);  convert_element_type_45 = None
    mul_516: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_212, mul_515);  squeeze_212 = mul_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_517: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_516, alias_35);  mul_516 = None
    sum_157: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_517, [3], True)
    mul_518: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_35, sum_157);  alias_35 = sum_157 = None
    sub_144: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_517, mul_518);  mul_517 = mul_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_519: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_144, 0.125);  sub_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_9: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_519, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1303: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_9, [1, 16, 1023, 512]);  index_put_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_37: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1303, 2, 1, 9223372036854775807);  view_1303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1304: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_37, [1, 16, 512, 1024]);  slice_scatter_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1305: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1304, [1, 16, 512, 1024, 1]);  view_1304 = None
    permute_1589: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1305, [1, 2, 4, 0, 3]);  view_1305 = None
    view_1306: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_1589, [16, 512, 1024]);  permute_1589 = None
    bmm_331: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_1590, view_1306);  permute_1590 = None
    bmm_332: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1306, permute_1591);  view_1306 = permute_1591 = None
    view_1307: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_331, [16, 64, 1, 1024, 1]);  bmm_331 = None
    permute_1592: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1307, [2, 0, 4, 3, 1]);  view_1307 = None
    view_1308: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_332, [16, 512, 64, 1, 1]);  bmm_332 = None
    permute_1593: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1308, [3, 0, 1, 4, 2]);  view_1308 = None
    permute_1594: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1592, [3, 0, 1, 4, 2]);  permute_1592 = None
    squeeze_213: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1594, 4);  permute_1594 = None
    permute_1595: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1593, [2, 0, 1, 4, 3]);  permute_1593 = None
    squeeze_214: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1595, 4);  permute_1595 = None
    sum_158: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_214, [0, 1], True)
    view_1309: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_158, [16, 64]);  sum_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1310: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_519, [1, 16, 512, 512, 1]);  mul_519 = None
    permute_1596: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1310, [1, 2, 4, 0, 3]);  view_1310 = None
    view_1311: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_1596, [16, 512, 512]);  permute_1596 = None
    bmm_333: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_1597, view_1311);  permute_1597 = None
    bmm_334: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1311, permute_1598);  view_1311 = permute_1598 = None
    view_1312: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_333, [16, 64, 1, 512, 1]);  bmm_333 = None
    permute_1599: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1312, [2, 0, 4, 3, 1]);  view_1312 = None
    view_1313: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_334, [16, 512, 64, 1, 1]);  bmm_334 = None
    permute_1600: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1313, [3, 0, 1, 4, 2]);  view_1313 = None
    permute_1601: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1599, [3, 0, 1, 4, 2]);  permute_1599 = None
    squeeze_215: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1601, 4);  permute_1601 = None
    permute_1602: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1600, [2, 0, 1, 4, 3]);  permute_1600 = None
    squeeze_216: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1602, 4);  permute_1602 = None
    sum_159: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_216, [0, 1], True)
    view_1314: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_159, [16, 64]);  sum_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_333: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_214, squeeze_216);  squeeze_214 = squeeze_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1315: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_213, [1024, 1, 16, 64, 1]);  squeeze_213 = None
    permute_1603: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1315, [0, 4, 1, 2, 3]);  view_1315 = None
    view_1316: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_1603, [1, 1024, 1024]);  permute_1603 = None
    bmm_335: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1316);  view_1316 = None
    view_1317: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_335, [1024, 1, 16, 64, 1]);  bmm_335 = None
    permute_1605: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1317, [4, 1, 2, 3, 0]);  view_1317 = None
    permute_1606: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1605, [4, 2, 3, 0, 1]);  permute_1605 = None
    squeeze_217: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1606, 4);  permute_1606 = None
    squeeze_218: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_217, 3);  squeeze_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1318: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_211, [512, 1, 16, 64, 1]);  squeeze_211 = None
    permute_1607: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1318, [0, 4, 1, 2, 3]);  view_1318 = None
    clone_107: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1607, memory_format = torch.contiguous_format);  permute_1607 = None
    view_1319: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_107, [1, 512, 1024]);  clone_107 = None
    bmm_336: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1608, view_1319)
    bmm_337: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1319, permute_1609);  view_1319 = permute_1609 = None
    view_1320: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_336, [1024, 1, 16, 64, 1]);  bmm_336 = None
    permute_1610: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1320, [4, 1, 2, 3, 0]);  view_1320 = None
    view_1321: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_337, [512, 1024, 1, 1, 1]);  bmm_337 = None
    permute_1611: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1321, [0, 2, 3, 4, 1]);  view_1321 = None
    permute_1612: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1610, [4, 2, 3, 0, 1]);  permute_1610 = None
    squeeze_219: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1612, 4);  permute_1612 = None
    squeeze_220: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_219, 3);  squeeze_219 = None
    permute_1613: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1611, [0, 1, 4, 2, 3]);  permute_1611 = None
    squeeze_221: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1613, 4);  permute_1613 = None
    squeeze_222: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_221, 3);  squeeze_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_334: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_511, squeeze_222);  mul_511 = squeeze_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1322: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_215, [512, 1, 16, 64, 1]);  squeeze_215 = None
    permute_1614: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1322, [0, 4, 1, 2, 3]);  view_1322 = None
    view_1323: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1614, [1, 512, 1024]);  permute_1614 = None
    bmm_338: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1608, view_1323)
    bmm_339: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1323, permute_1616);  view_1323 = permute_1616 = None
    view_1324: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_338, [1024, 1, 16, 64, 1]);  bmm_338 = None
    permute_1617: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1324, [4, 1, 2, 3, 0]);  view_1324 = None
    view_1325: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_339, [512, 1024, 1, 1, 1]);  bmm_339 = None
    permute_1618: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1325, [0, 2, 3, 4, 1]);  view_1325 = None
    permute_1619: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1617, [4, 2, 3, 0, 1]);  permute_1617 = None
    squeeze_223: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1619, 4);  permute_1619 = None
    squeeze_224: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_223, 3);  squeeze_223 = None
    permute_1620: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1618, [0, 1, 4, 2, 3]);  permute_1618 = None
    squeeze_225: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1620, 4);  permute_1620 = None
    squeeze_226: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_225, 3);  squeeze_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_335: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_334, squeeze_226);  add_334 = squeeze_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1326: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_333, [512, 1, 16, 64, 1]);  add_333 = None
    permute_1621: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1326, [0, 4, 1, 2, 3]);  view_1326 = None
    clone_108: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1621, memory_format = torch.contiguous_format);  permute_1621 = None
    view_1327: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_108, [1, 512, 1024]);  clone_108 = None
    bmm_340: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1608, view_1327);  permute_1608 = None
    bmm_341: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1327, permute_1623);  view_1327 = permute_1623 = None
    view_1328: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_340, [1024, 1, 16, 64, 1]);  bmm_340 = None
    permute_1624: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1328, [4, 1, 2, 3, 0]);  view_1328 = None
    view_1329: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_341, [512, 1024, 1, 1, 1]);  bmm_341 = None
    permute_1625: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1329, [0, 2, 3, 4, 1]);  view_1329 = None
    permute_1626: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1624, [4, 2, 3, 0, 1]);  permute_1624 = None
    squeeze_227: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1626, 4);  permute_1626 = None
    squeeze_228: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_227, 3);  squeeze_227 = None
    permute_1627: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1625, [0, 1, 4, 2, 3]);  permute_1625 = None
    squeeze_229: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1627, 4);  permute_1627 = None
    squeeze_230: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_229, 3);  squeeze_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_336: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_335, squeeze_230);  add_335 = squeeze_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_521: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_336, primals_280);  primals_280 = None
    mul_522: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_521, 1024)
    sum_160: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_521, [2], True)
    mul_523: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_521, mul_114);  mul_521 = None
    sum_161: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_523, [2], True);  mul_523 = None
    mul_524: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_114, sum_161);  sum_161 = None
    sub_146: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_522, sum_160);  mul_522 = sum_160 = None
    sub_147: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_146, mul_524);  sub_146 = mul_524 = None
    mul_525: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_47, sub_147);  div_47 = sub_147 = None
    mul_526: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_336, mul_114);  mul_114 = None
    sum_162: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_526, [0, 1]);  mul_526 = None
    sum_163: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_336, [0, 1]);  add_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_46: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_169, torch.float32);  getitem_169 = None
    mul_527: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 1.1111111111111112);  convert_element_type_46 = None
    mul_528: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_525, mul_527);  mul_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1330: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_528, [512, 1024]);  mul_528 = None
    mm_42: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1330, permute_1628);  permute_1628 = None
    permute_1629: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1330, [1, 0])
    mm_43: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1629, view_530);  permute_1629 = view_530 = None
    permute_1630: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_164: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1330, [0], True);  view_1330 = None
    view_1331: "f32[1024]" = torch.ops.aten.reshape.default(sum_164, [1024]);  sum_164 = None
    permute_1631: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1630, [1, 0]);  permute_1630 = None
    view_1332: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_42, [512, 1, 4096]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_47: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_167, torch.float32);  getitem_167 = None
    mul_529: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.1111111111111112);  convert_element_type_47 = None
    mul_530: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1332, mul_529);  view_1332 = mul_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_532: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_152, 0.5);  add_152 = None
    mul_533: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_529, view_529)
    mul_534: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_533, -0.5);  mul_533 = None
    exp_36: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_534);  mul_534 = None
    mul_535: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
    mul_536: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_529, mul_535);  view_529 = mul_535 = None
    add_338: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_532, mul_536);  mul_532 = mul_536 = None
    mul_537: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_530, add_338);  mul_530 = add_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1333: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_537, [512, 4096]);  mul_537 = None
    mm_44: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1333, permute_1632);  permute_1632 = None
    permute_1633: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1333, [1, 0])
    mm_45: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1633, view_528);  permute_1633 = view_528 = None
    permute_1634: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_165: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1333, [0], True);  view_1333 = None
    view_1334: "f32[4096]" = torch.ops.aten.reshape.default(sum_165, [4096]);  sum_165 = None
    permute_1635: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1634, [1, 0]);  permute_1634 = None
    view_1335: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_44, [512, 1, 1024]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_339: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_525, view_1335);  mul_525 = view_1335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_539: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_339, primals_274);  primals_274 = None
    mul_540: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_539, 1024)
    sum_166: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_539, [2], True)
    mul_541: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_539, mul_109);  mul_539 = None
    sum_167: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_541, [2], True);  mul_541 = None
    mul_542: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_109, sum_167);  sum_167 = None
    sub_149: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_540, sum_166);  mul_540 = sum_166 = None
    sub_150: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_149, mul_542);  sub_149 = mul_542 = None
    mul_543: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_48, sub_150);  div_48 = sub_150 = None
    mul_544: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_339, mul_109);  mul_109 = None
    sum_168: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_544, [0, 1]);  mul_544 = None
    sum_169: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_339, [0, 1]);  add_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_48: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_163, torch.float32);  getitem_163 = None
    mul_545: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 1.1111111111111112);  convert_element_type_48 = None
    mul_546: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_543, mul_545);  mul_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1336: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_546, [512, 1, 1024, 1, 1]);  mul_546 = None
    permute_1636: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1336, [0, 3, 4, 1, 2]);  view_1336 = None
    view_1337: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1636, [1, 512, 1024]);  permute_1636 = None
    bmm_342: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1637, view_1337);  permute_1637 = None
    bmm_343: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1337, permute_1638);  view_1337 = permute_1638 = None
    view_1338: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_342, [64, 16, 1, 1024, 1]);  bmm_342 = None
    permute_1639: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1338, [4, 2, 3, 0, 1]);  view_1338 = None
    view_1339: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_343, [512, 64, 16, 1, 1]);  bmm_343 = None
    permute_1640: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1339, [0, 3, 4, 1, 2]);  view_1339 = None
    permute_1641: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1639, [2, 4, 3, 0, 1]);  permute_1639 = None
    squeeze_231: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1641, 4);  permute_1641 = None
    squeeze_232: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_231, 3);  squeeze_231 = None
    permute_1642: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1640, [0, 1, 4, 3, 2]);  permute_1640 = None
    squeeze_233: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1642, 4);  permute_1642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1340: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_233, [512, 1, 16, 64, 1]);  squeeze_233 = None
    permute_1643: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1340, [2, 0, 4, 1, 3]);  view_1340 = None
    view_1341: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_1643, [16, 512, 64]);  permute_1643 = None
    bmm_344: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_1644, view_1341);  permute_1644 = None
    bmm_345: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1341, permute_1645);  view_1341 = permute_1645 = None
    view_1342: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_344, [16, 512, 1, 64, 1]);  bmm_344 = None
    permute_1646: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1342, [4, 2, 0, 3, 1]);  view_1342 = None
    view_1343: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_345, [16, 512, 512, 1, 1]);  bmm_345 = None
    permute_1647: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1343, [1, 3, 0, 4, 2]);  view_1343 = None
    permute_1648: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1646, [4, 1, 2, 3, 0]);  permute_1646 = None
    squeeze_234: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1648, 4);  permute_1648 = None
    permute_1649: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_1647, [1, 2, 0, 4, 3]);  permute_1647 = None
    squeeze_235: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_1649, 4);  permute_1649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_49: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_161, torch.float32);  getitem_161 = None
    mul_547: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 1.1111111111111112);  convert_element_type_49 = None
    mul_548: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_235, mul_547);  squeeze_235 = mul_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_549: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_548, alias_36);  mul_548 = None
    sum_170: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_549, [3], True)
    mul_550: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_36, sum_170);  alias_36 = sum_170 = None
    sub_151: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_549, mul_550);  mul_549 = mul_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_551: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_151, 0.125);  sub_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_10: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_551, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1344: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_10, [1, 16, 1023, 512]);  index_put_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_41: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1344, 2, 1, 9223372036854775807);  view_1344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1345: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_41, [1, 16, 512, 1024]);  slice_scatter_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1346: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1345, [1, 16, 512, 1024, 1]);  view_1345 = None
    permute_1650: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1346, [1, 2, 4, 0, 3]);  view_1346 = None
    view_1347: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_1650, [16, 512, 1024]);  permute_1650 = None
    bmm_346: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_1651, view_1347);  permute_1651 = None
    bmm_347: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1347, permute_1652);  view_1347 = permute_1652 = None
    view_1348: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_346, [16, 64, 1, 1024, 1]);  bmm_346 = None
    permute_1653: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1348, [2, 0, 4, 3, 1]);  view_1348 = None
    view_1349: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_347, [16, 512, 64, 1, 1]);  bmm_347 = None
    permute_1654: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1349, [3, 0, 1, 4, 2]);  view_1349 = None
    permute_1655: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1653, [3, 0, 1, 4, 2]);  permute_1653 = None
    squeeze_236: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1655, 4);  permute_1655 = None
    permute_1656: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1654, [2, 0, 1, 4, 3]);  permute_1654 = None
    squeeze_237: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1656, 4);  permute_1656 = None
    sum_171: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_237, [0, 1], True)
    view_1350: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_171, [16, 64]);  sum_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1351: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_551, [1, 16, 512, 512, 1]);  mul_551 = None
    permute_1657: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1351, [1, 2, 4, 0, 3]);  view_1351 = None
    view_1352: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_1657, [16, 512, 512]);  permute_1657 = None
    bmm_348: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_1658, view_1352);  permute_1658 = None
    bmm_349: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1352, permute_1659);  view_1352 = permute_1659 = None
    view_1353: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_348, [16, 64, 1, 512, 1]);  bmm_348 = None
    permute_1660: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1353, [2, 0, 4, 3, 1]);  view_1353 = None
    view_1354: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_349, [16, 512, 64, 1, 1]);  bmm_349 = None
    permute_1661: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1354, [3, 0, 1, 4, 2]);  view_1354 = None
    permute_1662: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1660, [3, 0, 1, 4, 2]);  permute_1660 = None
    squeeze_238: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1662, 4);  permute_1662 = None
    permute_1663: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1661, [2, 0, 1, 4, 3]);  permute_1661 = None
    squeeze_239: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1663, 4);  permute_1663 = None
    sum_172: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_239, [0, 1], True)
    view_1355: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_172, [16, 64]);  sum_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_340: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_237, squeeze_239);  squeeze_237 = squeeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1356: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_236, [1024, 1, 16, 64, 1]);  squeeze_236 = None
    permute_1664: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1356, [0, 4, 1, 2, 3]);  view_1356 = None
    view_1357: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_1664, [1, 1024, 1024]);  permute_1664 = None
    bmm_350: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1357);  view_1357 = None
    view_1358: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_350, [1024, 1, 16, 64, 1]);  bmm_350 = None
    permute_1666: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1358, [4, 1, 2, 3, 0]);  view_1358 = None
    permute_1667: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1666, [4, 2, 3, 0, 1]);  permute_1666 = None
    squeeze_240: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1667, 4);  permute_1667 = None
    squeeze_241: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_240, 3);  squeeze_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1359: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_234, [512, 1, 16, 64, 1]);  squeeze_234 = None
    permute_1668: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1359, [0, 4, 1, 2, 3]);  view_1359 = None
    clone_113: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1668, memory_format = torch.contiguous_format);  permute_1668 = None
    view_1360: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_113, [1, 512, 1024]);  clone_113 = None
    bmm_351: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1669, view_1360)
    bmm_352: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1360, permute_1670);  view_1360 = permute_1670 = None
    view_1361: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_351, [1024, 1, 16, 64, 1]);  bmm_351 = None
    permute_1671: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1361, [4, 1, 2, 3, 0]);  view_1361 = None
    view_1362: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_352, [512, 1024, 1, 1, 1]);  bmm_352 = None
    permute_1672: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1362, [0, 2, 3, 4, 1]);  view_1362 = None
    permute_1673: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1671, [4, 2, 3, 0, 1]);  permute_1671 = None
    squeeze_242: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1673, 4);  permute_1673 = None
    squeeze_243: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_242, 3);  squeeze_242 = None
    permute_1674: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1672, [0, 1, 4, 2, 3]);  permute_1672 = None
    squeeze_244: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1674, 4);  permute_1674 = None
    squeeze_245: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_244, 3);  squeeze_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_341: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_543, squeeze_245);  mul_543 = squeeze_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1363: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_238, [512, 1, 16, 64, 1]);  squeeze_238 = None
    permute_1675: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1363, [0, 4, 1, 2, 3]);  view_1363 = None
    view_1364: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1675, [1, 512, 1024]);  permute_1675 = None
    bmm_353: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1669, view_1364)
    bmm_354: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1364, permute_1677);  view_1364 = permute_1677 = None
    view_1365: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_353, [1024, 1, 16, 64, 1]);  bmm_353 = None
    permute_1678: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1365, [4, 1, 2, 3, 0]);  view_1365 = None
    view_1366: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_354, [512, 1024, 1, 1, 1]);  bmm_354 = None
    permute_1679: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1366, [0, 2, 3, 4, 1]);  view_1366 = None
    permute_1680: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1678, [4, 2, 3, 0, 1]);  permute_1678 = None
    squeeze_246: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1680, 4);  permute_1680 = None
    squeeze_247: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_246, 3);  squeeze_246 = None
    permute_1681: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1679, [0, 1, 4, 2, 3]);  permute_1679 = None
    squeeze_248: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1681, 4);  permute_1681 = None
    squeeze_249: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_248, 3);  squeeze_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_342: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_341, squeeze_249);  add_341 = squeeze_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1367: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_340, [512, 1, 16, 64, 1]);  add_340 = None
    permute_1682: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1367, [0, 4, 1, 2, 3]);  view_1367 = None
    clone_114: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1682, memory_format = torch.contiguous_format);  permute_1682 = None
    view_1368: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_114, [1, 512, 1024]);  clone_114 = None
    bmm_355: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1669, view_1368);  permute_1669 = None
    bmm_356: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1368, permute_1684);  view_1368 = permute_1684 = None
    view_1369: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_355, [1024, 1, 16, 64, 1]);  bmm_355 = None
    permute_1685: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1369, [4, 1, 2, 3, 0]);  view_1369 = None
    view_1370: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_356, [512, 1024, 1, 1, 1]);  bmm_356 = None
    permute_1686: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1370, [0, 2, 3, 4, 1]);  view_1370 = None
    permute_1687: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1685, [4, 2, 3, 0, 1]);  permute_1685 = None
    squeeze_250: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1687, 4);  permute_1687 = None
    squeeze_251: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_250, 3);  squeeze_250 = None
    permute_1688: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1686, [0, 1, 4, 2, 3]);  permute_1686 = None
    squeeze_252: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1688, 4);  permute_1688 = None
    squeeze_253: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_252, 3);  squeeze_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_343: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_342, squeeze_253);  add_342 = squeeze_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_553: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_343, primals_272);  primals_272 = None
    mul_554: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_553, 1024)
    sum_173: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_553, [2], True)
    mul_555: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_553, mul_106);  mul_553 = None
    sum_174: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_555, [2], True);  mul_555 = None
    mul_556: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_106, sum_174);  sum_174 = None
    sub_153: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_554, sum_173);  mul_554 = sum_173 = None
    sub_154: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_153, mul_556);  sub_153 = mul_556 = None
    mul_557: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_49, sub_154);  div_49 = sub_154 = None
    mul_558: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_343, mul_106);  mul_106 = None
    sum_175: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_558, [0, 1]);  mul_558 = None
    sum_176: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_343, [0, 1]);  add_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_50: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_157, torch.float32);  getitem_157 = None
    mul_559: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 1.1111111111111112);  convert_element_type_50 = None
    mul_560: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_557, mul_559);  mul_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1371: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_560, [512, 1024]);  mul_560 = None
    mm_46: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1371, permute_1689);  permute_1689 = None
    permute_1690: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1371, [1, 0])
    mm_47: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1690, view_492);  permute_1690 = view_492 = None
    permute_1691: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_177: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1371, [0], True);  view_1371 = None
    view_1372: "f32[1024]" = torch.ops.aten.reshape.default(sum_177, [1024]);  sum_177 = None
    permute_1692: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1691, [1, 0]);  permute_1691 = None
    view_1373: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_46, [512, 1, 4096]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_51: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_155, torch.float32);  getitem_155 = None
    mul_561: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_51, 1.1111111111111112);  convert_element_type_51 = None
    mul_562: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1373, mul_561);  view_1373 = mul_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_564: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_141, 0.5);  add_141 = None
    mul_565: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_491, view_491)
    mul_566: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_565, -0.5);  mul_565 = None
    exp_37: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_566);  mul_566 = None
    mul_567: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
    mul_568: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_491, mul_567);  view_491 = mul_567 = None
    add_345: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_564, mul_568);  mul_564 = mul_568 = None
    mul_569: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_562, add_345);  mul_562 = add_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1374: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_569, [512, 4096]);  mul_569 = None
    mm_48: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1374, permute_1693);  permute_1693 = None
    permute_1694: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1374, [1, 0])
    mm_49: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1694, view_490);  permute_1694 = view_490 = None
    permute_1695: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_178: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1374, [0], True);  view_1374 = None
    view_1375: "f32[4096]" = torch.ops.aten.reshape.default(sum_178, [4096]);  sum_178 = None
    permute_1696: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1695, [1, 0]);  permute_1695 = None
    view_1376: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_48, [512, 1, 1024]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_346: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_557, view_1376);  mul_557 = view_1376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_571: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_346, primals_266);  primals_266 = None
    mul_572: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_571, 1024)
    sum_179: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_571, [2], True)
    mul_573: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_571, mul_101);  mul_571 = None
    sum_180: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_573, [2], True);  mul_573 = None
    mul_574: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_101, sum_180);  sum_180 = None
    sub_156: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_572, sum_179);  mul_572 = sum_179 = None
    sub_157: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_156, mul_574);  sub_156 = mul_574 = None
    mul_575: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_50, sub_157);  div_50 = sub_157 = None
    mul_576: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_346, mul_101);  mul_101 = None
    sum_181: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_576, [0, 1]);  mul_576 = None
    sum_182: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_346, [0, 1]);  add_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_52: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_151, torch.float32);  getitem_151 = None
    mul_577: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_52, 1.1111111111111112);  convert_element_type_52 = None
    mul_578: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_575, mul_577);  mul_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1377: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_578, [512, 1, 1024, 1, 1]);  mul_578 = None
    permute_1697: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1377, [0, 3, 4, 1, 2]);  view_1377 = None
    view_1378: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1697, [1, 512, 1024]);  permute_1697 = None
    bmm_357: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1698, view_1378);  permute_1698 = None
    bmm_358: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1378, permute_1699);  view_1378 = permute_1699 = None
    view_1379: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_357, [64, 16, 1, 1024, 1]);  bmm_357 = None
    permute_1700: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1379, [4, 2, 3, 0, 1]);  view_1379 = None
    view_1380: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_358, [512, 64, 16, 1, 1]);  bmm_358 = None
    permute_1701: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1380, [0, 3, 4, 1, 2]);  view_1380 = None
    permute_1702: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1700, [2, 4, 3, 0, 1]);  permute_1700 = None
    squeeze_254: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1702, 4);  permute_1702 = None
    squeeze_255: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_254, 3);  squeeze_254 = None
    permute_1703: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1701, [0, 1, 4, 3, 2]);  permute_1701 = None
    squeeze_256: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1703, 4);  permute_1703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1381: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_256, [512, 1, 16, 64, 1]);  squeeze_256 = None
    permute_1704: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1381, [2, 0, 4, 1, 3]);  view_1381 = None
    view_1382: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_1704, [16, 512, 64]);  permute_1704 = None
    bmm_359: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_1705, view_1382);  permute_1705 = None
    bmm_360: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1382, permute_1706);  view_1382 = permute_1706 = None
    view_1383: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_359, [16, 512, 1, 64, 1]);  bmm_359 = None
    permute_1707: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1383, [4, 2, 0, 3, 1]);  view_1383 = None
    view_1384: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_360, [16, 512, 512, 1, 1]);  bmm_360 = None
    permute_1708: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1384, [1, 3, 0, 4, 2]);  view_1384 = None
    permute_1709: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1707, [4, 1, 2, 3, 0]);  permute_1707 = None
    squeeze_257: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1709, 4);  permute_1709 = None
    permute_1710: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_1708, [1, 2, 0, 4, 3]);  permute_1708 = None
    squeeze_258: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_1710, 4);  permute_1710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_53: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_149, torch.float32);  getitem_149 = None
    mul_579: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 1.1111111111111112);  convert_element_type_53 = None
    mul_580: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_258, mul_579);  squeeze_258 = mul_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_581: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_580, alias_37);  mul_580 = None
    sum_183: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_581, [3], True)
    mul_582: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_37, sum_183);  alias_37 = sum_183 = None
    sub_158: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_581, mul_582);  mul_581 = mul_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_583: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_158, 0.125);  sub_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_11: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_583, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1385: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_11, [1, 16, 1023, 512]);  index_put_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_45: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1385, 2, 1, 9223372036854775807);  view_1385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1386: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_45, [1, 16, 512, 1024]);  slice_scatter_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1387: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1386, [1, 16, 512, 1024, 1]);  view_1386 = None
    permute_1711: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1387, [1, 2, 4, 0, 3]);  view_1387 = None
    view_1388: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_1711, [16, 512, 1024]);  permute_1711 = None
    bmm_361: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_1712, view_1388);  permute_1712 = None
    bmm_362: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1388, permute_1713);  view_1388 = permute_1713 = None
    view_1389: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_361, [16, 64, 1, 1024, 1]);  bmm_361 = None
    permute_1714: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1389, [2, 0, 4, 3, 1]);  view_1389 = None
    view_1390: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_362, [16, 512, 64, 1, 1]);  bmm_362 = None
    permute_1715: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1390, [3, 0, 1, 4, 2]);  view_1390 = None
    permute_1716: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1714, [3, 0, 1, 4, 2]);  permute_1714 = None
    squeeze_259: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1716, 4);  permute_1716 = None
    permute_1717: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1715, [2, 0, 1, 4, 3]);  permute_1715 = None
    squeeze_260: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1717, 4);  permute_1717 = None
    sum_184: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_260, [0, 1], True)
    view_1391: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_184, [16, 64]);  sum_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1392: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_583, [1, 16, 512, 512, 1]);  mul_583 = None
    permute_1718: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1392, [1, 2, 4, 0, 3]);  view_1392 = None
    view_1393: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_1718, [16, 512, 512]);  permute_1718 = None
    bmm_363: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_1719, view_1393);  permute_1719 = None
    bmm_364: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1393, permute_1720);  view_1393 = permute_1720 = None
    view_1394: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_363, [16, 64, 1, 512, 1]);  bmm_363 = None
    permute_1721: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1394, [2, 0, 4, 3, 1]);  view_1394 = None
    view_1395: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_364, [16, 512, 64, 1, 1]);  bmm_364 = None
    permute_1722: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1395, [3, 0, 1, 4, 2]);  view_1395 = None
    permute_1723: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1721, [3, 0, 1, 4, 2]);  permute_1721 = None
    squeeze_261: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1723, 4);  permute_1723 = None
    permute_1724: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1722, [2, 0, 1, 4, 3]);  permute_1722 = None
    squeeze_262: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1724, 4);  permute_1724 = None
    sum_185: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_262, [0, 1], True)
    view_1396: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_185, [16, 64]);  sum_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_347: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_260, squeeze_262);  squeeze_260 = squeeze_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1397: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_259, [1024, 1, 16, 64, 1]);  squeeze_259 = None
    permute_1725: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1397, [0, 4, 1, 2, 3]);  view_1397 = None
    view_1398: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_1725, [1, 1024, 1024]);  permute_1725 = None
    bmm_365: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1398);  view_1398 = None
    view_1399: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_365, [1024, 1, 16, 64, 1]);  bmm_365 = None
    permute_1727: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1399, [4, 1, 2, 3, 0]);  view_1399 = None
    permute_1728: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1727, [4, 2, 3, 0, 1]);  permute_1727 = None
    squeeze_263: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1728, 4);  permute_1728 = None
    squeeze_264: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_263, 3);  squeeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1400: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_257, [512, 1, 16, 64, 1]);  squeeze_257 = None
    permute_1729: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1400, [0, 4, 1, 2, 3]);  view_1400 = None
    clone_119: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1729, memory_format = torch.contiguous_format);  permute_1729 = None
    view_1401: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_119, [1, 512, 1024]);  clone_119 = None
    bmm_366: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1730, view_1401)
    bmm_367: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1401, permute_1731);  view_1401 = permute_1731 = None
    view_1402: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_366, [1024, 1, 16, 64, 1]);  bmm_366 = None
    permute_1732: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1402, [4, 1, 2, 3, 0]);  view_1402 = None
    view_1403: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_367, [512, 1024, 1, 1, 1]);  bmm_367 = None
    permute_1733: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1403, [0, 2, 3, 4, 1]);  view_1403 = None
    permute_1734: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1732, [4, 2, 3, 0, 1]);  permute_1732 = None
    squeeze_265: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1734, 4);  permute_1734 = None
    squeeze_266: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_265, 3);  squeeze_265 = None
    permute_1735: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1733, [0, 1, 4, 2, 3]);  permute_1733 = None
    squeeze_267: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1735, 4);  permute_1735 = None
    squeeze_268: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_267, 3);  squeeze_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_348: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_575, squeeze_268);  mul_575 = squeeze_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1404: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_261, [512, 1, 16, 64, 1]);  squeeze_261 = None
    permute_1736: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1404, [0, 4, 1, 2, 3]);  view_1404 = None
    view_1405: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1736, [1, 512, 1024]);  permute_1736 = None
    bmm_368: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1730, view_1405)
    bmm_369: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1405, permute_1738);  view_1405 = permute_1738 = None
    view_1406: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_368, [1024, 1, 16, 64, 1]);  bmm_368 = None
    permute_1739: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1406, [4, 1, 2, 3, 0]);  view_1406 = None
    view_1407: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_369, [512, 1024, 1, 1, 1]);  bmm_369 = None
    permute_1740: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1407, [0, 2, 3, 4, 1]);  view_1407 = None
    permute_1741: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1739, [4, 2, 3, 0, 1]);  permute_1739 = None
    squeeze_269: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1741, 4);  permute_1741 = None
    squeeze_270: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_269, 3);  squeeze_269 = None
    permute_1742: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1740, [0, 1, 4, 2, 3]);  permute_1740 = None
    squeeze_271: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1742, 4);  permute_1742 = None
    squeeze_272: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_271, 3);  squeeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_349: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_348, squeeze_272);  add_348 = squeeze_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1408: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_347, [512, 1, 16, 64, 1]);  add_347 = None
    permute_1743: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1408, [0, 4, 1, 2, 3]);  view_1408 = None
    clone_120: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1743, memory_format = torch.contiguous_format);  permute_1743 = None
    view_1409: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_120, [1, 512, 1024]);  clone_120 = None
    bmm_370: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1730, view_1409);  permute_1730 = None
    bmm_371: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1409, permute_1745);  view_1409 = permute_1745 = None
    view_1410: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_370, [1024, 1, 16, 64, 1]);  bmm_370 = None
    permute_1746: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1410, [4, 1, 2, 3, 0]);  view_1410 = None
    view_1411: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_371, [512, 1024, 1, 1, 1]);  bmm_371 = None
    permute_1747: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1411, [0, 2, 3, 4, 1]);  view_1411 = None
    permute_1748: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1746, [4, 2, 3, 0, 1]);  permute_1746 = None
    squeeze_273: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1748, 4);  permute_1748 = None
    squeeze_274: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_273, 3);  squeeze_273 = None
    permute_1749: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1747, [0, 1, 4, 2, 3]);  permute_1747 = None
    squeeze_275: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1749, 4);  permute_1749 = None
    squeeze_276: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_275, 3);  squeeze_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_350: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_349, squeeze_276);  add_349 = squeeze_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_585: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_350, primals_264);  primals_264 = None
    mul_586: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_585, 1024)
    sum_186: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_585, [2], True)
    mul_587: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_585, mul_98);  mul_585 = None
    sum_187: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_587, [2], True);  mul_587 = None
    mul_588: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_98, sum_187);  sum_187 = None
    sub_160: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_586, sum_186);  mul_586 = sum_186 = None
    sub_161: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_160, mul_588);  sub_160 = mul_588 = None
    mul_589: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_51, sub_161);  div_51 = sub_161 = None
    mul_590: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_350, mul_98);  mul_98 = None
    sum_188: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_590, [0, 1]);  mul_590 = None
    sum_189: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_350, [0, 1]);  add_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_54: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_145, torch.float32);  getitem_145 = None
    mul_591: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 1.1111111111111112);  convert_element_type_54 = None
    mul_592: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_589, mul_591);  mul_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1412: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_592, [512, 1024]);  mul_592 = None
    mm_50: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1412, permute_1750);  permute_1750 = None
    permute_1751: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1412, [1, 0])
    mm_51: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1751, view_454);  permute_1751 = view_454 = None
    permute_1752: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_190: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1412, [0], True);  view_1412 = None
    view_1413: "f32[1024]" = torch.ops.aten.reshape.default(sum_190, [1024]);  sum_190 = None
    permute_1753: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1752, [1, 0]);  permute_1752 = None
    view_1414: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_50, [512, 1, 4096]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_55: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_143, torch.float32);  getitem_143 = None
    mul_593: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 1.1111111111111112);  convert_element_type_55 = None
    mul_594: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1414, mul_593);  view_1414 = mul_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_596: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_130, 0.5);  add_130 = None
    mul_597: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_453, view_453)
    mul_598: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_597, -0.5);  mul_597 = None
    exp_38: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_598);  mul_598 = None
    mul_599: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
    mul_600: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_453, mul_599);  view_453 = mul_599 = None
    add_352: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_596, mul_600);  mul_596 = mul_600 = None
    mul_601: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_594, add_352);  mul_594 = add_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1415: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_601, [512, 4096]);  mul_601 = None
    mm_52: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1415, permute_1754);  permute_1754 = None
    permute_1755: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1415, [1, 0])
    mm_53: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1755, view_452);  permute_1755 = view_452 = None
    permute_1756: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_191: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1415, [0], True);  view_1415 = None
    view_1416: "f32[4096]" = torch.ops.aten.reshape.default(sum_191, [4096]);  sum_191 = None
    permute_1757: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1756, [1, 0]);  permute_1756 = None
    view_1417: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_52, [512, 1, 1024]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_353: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_589, view_1417);  mul_589 = view_1417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_603: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_353, primals_258);  primals_258 = None
    mul_604: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_603, 1024)
    sum_192: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_603, [2], True)
    mul_605: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_603, mul_93);  mul_603 = None
    sum_193: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_605, [2], True);  mul_605 = None
    mul_606: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_93, sum_193);  sum_193 = None
    sub_163: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_604, sum_192);  mul_604 = sum_192 = None
    sub_164: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_163, mul_606);  sub_163 = mul_606 = None
    mul_607: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_52, sub_164);  div_52 = sub_164 = None
    mul_608: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_353, mul_93);  mul_93 = None
    sum_194: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 1]);  mul_608 = None
    sum_195: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_353, [0, 1]);  add_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_56: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_139, torch.float32);  getitem_139 = None
    mul_609: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_56, 1.1111111111111112);  convert_element_type_56 = None
    mul_610: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_607, mul_609);  mul_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1418: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_610, [512, 1, 1024, 1, 1]);  mul_610 = None
    permute_1758: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1418, [0, 3, 4, 1, 2]);  view_1418 = None
    view_1419: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1758, [1, 512, 1024]);  permute_1758 = None
    bmm_372: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1759, view_1419);  permute_1759 = None
    bmm_373: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1419, permute_1760);  view_1419 = permute_1760 = None
    view_1420: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_372, [64, 16, 1, 1024, 1]);  bmm_372 = None
    permute_1761: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1420, [4, 2, 3, 0, 1]);  view_1420 = None
    view_1421: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_373, [512, 64, 16, 1, 1]);  bmm_373 = None
    permute_1762: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1421, [0, 3, 4, 1, 2]);  view_1421 = None
    permute_1763: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1761, [2, 4, 3, 0, 1]);  permute_1761 = None
    squeeze_277: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1763, 4);  permute_1763 = None
    squeeze_278: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_277, 3);  squeeze_277 = None
    permute_1764: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1762, [0, 1, 4, 3, 2]);  permute_1762 = None
    squeeze_279: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1764, 4);  permute_1764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1422: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_279, [512, 1, 16, 64, 1]);  squeeze_279 = None
    permute_1765: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1422, [2, 0, 4, 1, 3]);  view_1422 = None
    view_1423: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_1765, [16, 512, 64]);  permute_1765 = None
    bmm_374: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_1766, view_1423);  permute_1766 = None
    bmm_375: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1423, permute_1767);  view_1423 = permute_1767 = None
    view_1424: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_374, [16, 512, 1, 64, 1]);  bmm_374 = None
    permute_1768: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1424, [4, 2, 0, 3, 1]);  view_1424 = None
    view_1425: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_375, [16, 512, 512, 1, 1]);  bmm_375 = None
    permute_1769: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1425, [1, 3, 0, 4, 2]);  view_1425 = None
    permute_1770: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1768, [4, 1, 2, 3, 0]);  permute_1768 = None
    squeeze_280: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1770, 4);  permute_1770 = None
    permute_1771: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_1769, [1, 2, 0, 4, 3]);  permute_1769 = None
    squeeze_281: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_1771, 4);  permute_1771 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_57: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_137, torch.float32);  getitem_137 = None
    mul_611: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_57, 1.1111111111111112);  convert_element_type_57 = None
    mul_612: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_281, mul_611);  squeeze_281 = mul_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_613: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_612, alias_38);  mul_612 = None
    sum_196: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_613, [3], True)
    mul_614: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_38, sum_196);  alias_38 = sum_196 = None
    sub_165: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_613, mul_614);  mul_613 = mul_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_615: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_165, 0.125);  sub_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_12: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_615, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1426: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_12, [1, 16, 1023, 512]);  index_put_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_49: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1426, 2, 1, 9223372036854775807);  view_1426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1427: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_49, [1, 16, 512, 1024]);  slice_scatter_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1428: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1427, [1, 16, 512, 1024, 1]);  view_1427 = None
    permute_1772: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1428, [1, 2, 4, 0, 3]);  view_1428 = None
    view_1429: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_1772, [16, 512, 1024]);  permute_1772 = None
    bmm_376: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_1773, view_1429);  permute_1773 = None
    bmm_377: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1429, permute_1774);  view_1429 = permute_1774 = None
    view_1430: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_376, [16, 64, 1, 1024, 1]);  bmm_376 = None
    permute_1775: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1430, [2, 0, 4, 3, 1]);  view_1430 = None
    view_1431: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_377, [16, 512, 64, 1, 1]);  bmm_377 = None
    permute_1776: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1431, [3, 0, 1, 4, 2]);  view_1431 = None
    permute_1777: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1775, [3, 0, 1, 4, 2]);  permute_1775 = None
    squeeze_282: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1777, 4);  permute_1777 = None
    permute_1778: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1776, [2, 0, 1, 4, 3]);  permute_1776 = None
    squeeze_283: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1778, 4);  permute_1778 = None
    sum_197: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_283, [0, 1], True)
    view_1432: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_197, [16, 64]);  sum_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1433: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_615, [1, 16, 512, 512, 1]);  mul_615 = None
    permute_1779: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1433, [1, 2, 4, 0, 3]);  view_1433 = None
    view_1434: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_1779, [16, 512, 512]);  permute_1779 = None
    bmm_378: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_1780, view_1434);  permute_1780 = None
    bmm_379: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1434, permute_1781);  view_1434 = permute_1781 = None
    view_1435: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_378, [16, 64, 1, 512, 1]);  bmm_378 = None
    permute_1782: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1435, [2, 0, 4, 3, 1]);  view_1435 = None
    view_1436: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_379, [16, 512, 64, 1, 1]);  bmm_379 = None
    permute_1783: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1436, [3, 0, 1, 4, 2]);  view_1436 = None
    permute_1784: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1782, [3, 0, 1, 4, 2]);  permute_1782 = None
    squeeze_284: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1784, 4);  permute_1784 = None
    permute_1785: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1783, [2, 0, 1, 4, 3]);  permute_1783 = None
    squeeze_285: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1785, 4);  permute_1785 = None
    sum_198: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_285, [0, 1], True)
    view_1437: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_198, [16, 64]);  sum_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_354: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_283, squeeze_285);  squeeze_283 = squeeze_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1438: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_282, [1024, 1, 16, 64, 1]);  squeeze_282 = None
    permute_1786: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1438, [0, 4, 1, 2, 3]);  view_1438 = None
    view_1439: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_1786, [1, 1024, 1024]);  permute_1786 = None
    bmm_380: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1439);  view_1439 = None
    view_1440: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_380, [1024, 1, 16, 64, 1]);  bmm_380 = None
    permute_1788: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1440, [4, 1, 2, 3, 0]);  view_1440 = None
    permute_1789: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1788, [4, 2, 3, 0, 1]);  permute_1788 = None
    squeeze_286: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1789, 4);  permute_1789 = None
    squeeze_287: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_286, 3);  squeeze_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1441: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_280, [512, 1, 16, 64, 1]);  squeeze_280 = None
    permute_1790: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1441, [0, 4, 1, 2, 3]);  view_1441 = None
    clone_125: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1790, memory_format = torch.contiguous_format);  permute_1790 = None
    view_1442: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_125, [1, 512, 1024]);  clone_125 = None
    bmm_381: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1791, view_1442)
    bmm_382: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1442, permute_1792);  view_1442 = permute_1792 = None
    view_1443: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_381, [1024, 1, 16, 64, 1]);  bmm_381 = None
    permute_1793: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1443, [4, 1, 2, 3, 0]);  view_1443 = None
    view_1444: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_382, [512, 1024, 1, 1, 1]);  bmm_382 = None
    permute_1794: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1444, [0, 2, 3, 4, 1]);  view_1444 = None
    permute_1795: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1793, [4, 2, 3, 0, 1]);  permute_1793 = None
    squeeze_288: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1795, 4);  permute_1795 = None
    squeeze_289: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_288, 3);  squeeze_288 = None
    permute_1796: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1794, [0, 1, 4, 2, 3]);  permute_1794 = None
    squeeze_290: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1796, 4);  permute_1796 = None
    squeeze_291: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_290, 3);  squeeze_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_355: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_607, squeeze_291);  mul_607 = squeeze_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1445: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_284, [512, 1, 16, 64, 1]);  squeeze_284 = None
    permute_1797: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1445, [0, 4, 1, 2, 3]);  view_1445 = None
    view_1446: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1797, [1, 512, 1024]);  permute_1797 = None
    bmm_383: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1791, view_1446)
    bmm_384: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1446, permute_1799);  view_1446 = permute_1799 = None
    view_1447: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_383, [1024, 1, 16, 64, 1]);  bmm_383 = None
    permute_1800: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1447, [4, 1, 2, 3, 0]);  view_1447 = None
    view_1448: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_384, [512, 1024, 1, 1, 1]);  bmm_384 = None
    permute_1801: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1448, [0, 2, 3, 4, 1]);  view_1448 = None
    permute_1802: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1800, [4, 2, 3, 0, 1]);  permute_1800 = None
    squeeze_292: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1802, 4);  permute_1802 = None
    squeeze_293: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_292, 3);  squeeze_292 = None
    permute_1803: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1801, [0, 1, 4, 2, 3]);  permute_1801 = None
    squeeze_294: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1803, 4);  permute_1803 = None
    squeeze_295: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_294, 3);  squeeze_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_356: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_355, squeeze_295);  add_355 = squeeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1449: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_354, [512, 1, 16, 64, 1]);  add_354 = None
    permute_1804: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1449, [0, 4, 1, 2, 3]);  view_1449 = None
    clone_126: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1804, memory_format = torch.contiguous_format);  permute_1804 = None
    view_1450: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_126, [1, 512, 1024]);  clone_126 = None
    bmm_385: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1791, view_1450);  permute_1791 = None
    bmm_386: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1450, permute_1806);  view_1450 = permute_1806 = None
    view_1451: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_385, [1024, 1, 16, 64, 1]);  bmm_385 = None
    permute_1807: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1451, [4, 1, 2, 3, 0]);  view_1451 = None
    view_1452: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_386, [512, 1024, 1, 1, 1]);  bmm_386 = None
    permute_1808: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1452, [0, 2, 3, 4, 1]);  view_1452 = None
    permute_1809: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1807, [4, 2, 3, 0, 1]);  permute_1807 = None
    squeeze_296: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1809, 4);  permute_1809 = None
    squeeze_297: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_296, 3);  squeeze_296 = None
    permute_1810: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1808, [0, 1, 4, 2, 3]);  permute_1808 = None
    squeeze_298: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1810, 4);  permute_1810 = None
    squeeze_299: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_298, 3);  squeeze_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_357: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_356, squeeze_299);  add_356 = squeeze_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_617: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_357, primals_256);  primals_256 = None
    mul_618: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_617, 1024)
    sum_199: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_617, [2], True)
    mul_619: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_617, mul_90);  mul_617 = None
    sum_200: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_619, [2], True);  mul_619 = None
    mul_620: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_90, sum_200);  sum_200 = None
    sub_167: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_618, sum_199);  mul_618 = sum_199 = None
    sub_168: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_167, mul_620);  sub_167 = mul_620 = None
    mul_621: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_53, sub_168);  div_53 = sub_168 = None
    mul_622: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_357, mul_90);  mul_90 = None
    sum_201: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_622, [0, 1]);  mul_622 = None
    sum_202: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_357, [0, 1]);  add_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_58: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_133, torch.float32);  getitem_133 = None
    mul_623: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 1.1111111111111112);  convert_element_type_58 = None
    mul_624: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_621, mul_623);  mul_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1453: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_624, [512, 1024]);  mul_624 = None
    mm_54: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1453, permute_1811);  permute_1811 = None
    permute_1812: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1453, [1, 0])
    mm_55: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1812, view_416);  permute_1812 = view_416 = None
    permute_1813: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_203: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1453, [0], True);  view_1453 = None
    view_1454: "f32[1024]" = torch.ops.aten.reshape.default(sum_203, [1024]);  sum_203 = None
    permute_1814: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1813, [1, 0]);  permute_1813 = None
    view_1455: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_54, [512, 1, 4096]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_59: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_131, torch.float32);  getitem_131 = None
    mul_625: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 1.1111111111111112);  convert_element_type_59 = None
    mul_626: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1455, mul_625);  view_1455 = mul_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_628: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_119, 0.5);  add_119 = None
    mul_629: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_415, view_415)
    mul_630: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_629, -0.5);  mul_629 = None
    exp_39: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_630);  mul_630 = None
    mul_631: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
    mul_632: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_415, mul_631);  view_415 = mul_631 = None
    add_359: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_628, mul_632);  mul_628 = mul_632 = None
    mul_633: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_626, add_359);  mul_626 = add_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1456: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_633, [512, 4096]);  mul_633 = None
    mm_56: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1456, permute_1815);  permute_1815 = None
    permute_1816: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1456, [1, 0])
    mm_57: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1816, view_414);  permute_1816 = view_414 = None
    permute_1817: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_204: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1456, [0], True);  view_1456 = None
    view_1457: "f32[4096]" = torch.ops.aten.reshape.default(sum_204, [4096]);  sum_204 = None
    permute_1818: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1817, [1, 0]);  permute_1817 = None
    view_1458: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_56, [512, 1, 1024]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_360: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_621, view_1458);  mul_621 = view_1458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_635: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_360, primals_250);  primals_250 = None
    mul_636: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_635, 1024)
    sum_205: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_635, [2], True)
    mul_637: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_635, mul_85);  mul_635 = None
    sum_206: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_637, [2], True);  mul_637 = None
    mul_638: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_85, sum_206);  sum_206 = None
    sub_170: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_636, sum_205);  mul_636 = sum_205 = None
    sub_171: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_170, mul_638);  sub_170 = mul_638 = None
    mul_639: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_54, sub_171);  div_54 = sub_171 = None
    mul_640: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_360, mul_85);  mul_85 = None
    sum_207: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 1]);  mul_640 = None
    sum_208: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_360, [0, 1]);  add_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_60: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_127, torch.float32);  getitem_127 = None
    mul_641: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 1.1111111111111112);  convert_element_type_60 = None
    mul_642: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_639, mul_641);  mul_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1459: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_642, [512, 1, 1024, 1, 1]);  mul_642 = None
    permute_1819: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1459, [0, 3, 4, 1, 2]);  view_1459 = None
    view_1460: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1819, [1, 512, 1024]);  permute_1819 = None
    bmm_387: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1820, view_1460);  permute_1820 = None
    bmm_388: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1460, permute_1821);  view_1460 = permute_1821 = None
    view_1461: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_387, [64, 16, 1, 1024, 1]);  bmm_387 = None
    permute_1822: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1461, [4, 2, 3, 0, 1]);  view_1461 = None
    view_1462: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_388, [512, 64, 16, 1, 1]);  bmm_388 = None
    permute_1823: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1462, [0, 3, 4, 1, 2]);  view_1462 = None
    permute_1824: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1822, [2, 4, 3, 0, 1]);  permute_1822 = None
    squeeze_300: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1824, 4);  permute_1824 = None
    squeeze_301: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_300, 3);  squeeze_300 = None
    permute_1825: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1823, [0, 1, 4, 3, 2]);  permute_1823 = None
    squeeze_302: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1825, 4);  permute_1825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1463: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_302, [512, 1, 16, 64, 1]);  squeeze_302 = None
    permute_1826: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1463, [2, 0, 4, 1, 3]);  view_1463 = None
    view_1464: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_1826, [16, 512, 64]);  permute_1826 = None
    bmm_389: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_1827, view_1464);  permute_1827 = None
    bmm_390: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1464, permute_1828);  view_1464 = permute_1828 = None
    view_1465: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_389, [16, 512, 1, 64, 1]);  bmm_389 = None
    permute_1829: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1465, [4, 2, 0, 3, 1]);  view_1465 = None
    view_1466: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_390, [16, 512, 512, 1, 1]);  bmm_390 = None
    permute_1830: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1466, [1, 3, 0, 4, 2]);  view_1466 = None
    permute_1831: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1829, [4, 1, 2, 3, 0]);  permute_1829 = None
    squeeze_303: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1831, 4);  permute_1831 = None
    permute_1832: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_1830, [1, 2, 0, 4, 3]);  permute_1830 = None
    squeeze_304: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_1832, 4);  permute_1832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_61: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_125, torch.float32);  getitem_125 = None
    mul_643: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_61, 1.1111111111111112);  convert_element_type_61 = None
    mul_644: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_304, mul_643);  squeeze_304 = mul_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_645: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_644, alias_39);  mul_644 = None
    sum_209: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_645, [3], True)
    mul_646: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_39, sum_209);  alias_39 = sum_209 = None
    sub_172: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_645, mul_646);  mul_645 = mul_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_647: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_172, 0.125);  sub_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_13: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_647, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1467: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_13, [1, 16, 1023, 512]);  index_put_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_53: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1467, 2, 1, 9223372036854775807);  view_1467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1468: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_53, [1, 16, 512, 1024]);  slice_scatter_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1469: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1468, [1, 16, 512, 1024, 1]);  view_1468 = None
    permute_1833: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1469, [1, 2, 4, 0, 3]);  view_1469 = None
    view_1470: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_1833, [16, 512, 1024]);  permute_1833 = None
    bmm_391: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_1834, view_1470);  permute_1834 = None
    bmm_392: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1470, permute_1835);  view_1470 = permute_1835 = None
    view_1471: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_391, [16, 64, 1, 1024, 1]);  bmm_391 = None
    permute_1836: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1471, [2, 0, 4, 3, 1]);  view_1471 = None
    view_1472: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_392, [16, 512, 64, 1, 1]);  bmm_392 = None
    permute_1837: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1472, [3, 0, 1, 4, 2]);  view_1472 = None
    permute_1838: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1836, [3, 0, 1, 4, 2]);  permute_1836 = None
    squeeze_305: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1838, 4);  permute_1838 = None
    permute_1839: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1837, [2, 0, 1, 4, 3]);  permute_1837 = None
    squeeze_306: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1839, 4);  permute_1839 = None
    sum_210: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_306, [0, 1], True)
    view_1473: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_210, [16, 64]);  sum_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1474: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_647, [1, 16, 512, 512, 1]);  mul_647 = None
    permute_1840: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1474, [1, 2, 4, 0, 3]);  view_1474 = None
    view_1475: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_1840, [16, 512, 512]);  permute_1840 = None
    bmm_393: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_1841, view_1475);  permute_1841 = None
    bmm_394: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1475, permute_1842);  view_1475 = permute_1842 = None
    view_1476: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_393, [16, 64, 1, 512, 1]);  bmm_393 = None
    permute_1843: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1476, [2, 0, 4, 3, 1]);  view_1476 = None
    view_1477: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_394, [16, 512, 64, 1, 1]);  bmm_394 = None
    permute_1844: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1477, [3, 0, 1, 4, 2]);  view_1477 = None
    permute_1845: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1843, [3, 0, 1, 4, 2]);  permute_1843 = None
    squeeze_307: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1845, 4);  permute_1845 = None
    permute_1846: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1844, [2, 0, 1, 4, 3]);  permute_1844 = None
    squeeze_308: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1846, 4);  permute_1846 = None
    sum_211: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_308, [0, 1], True)
    view_1478: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_211, [16, 64]);  sum_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_361: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_306, squeeze_308);  squeeze_306 = squeeze_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1479: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_305, [1024, 1, 16, 64, 1]);  squeeze_305 = None
    permute_1847: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1479, [0, 4, 1, 2, 3]);  view_1479 = None
    view_1480: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_1847, [1, 1024, 1024]);  permute_1847 = None
    bmm_395: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1480);  view_1480 = None
    view_1481: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_395, [1024, 1, 16, 64, 1]);  bmm_395 = None
    permute_1849: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1481, [4, 1, 2, 3, 0]);  view_1481 = None
    permute_1850: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1849, [4, 2, 3, 0, 1]);  permute_1849 = None
    squeeze_309: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1850, 4);  permute_1850 = None
    squeeze_310: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_309, 3);  squeeze_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1482: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_303, [512, 1, 16, 64, 1]);  squeeze_303 = None
    permute_1851: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1482, [0, 4, 1, 2, 3]);  view_1482 = None
    clone_131: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1851, memory_format = torch.contiguous_format);  permute_1851 = None
    view_1483: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_131, [1, 512, 1024]);  clone_131 = None
    bmm_396: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1852, view_1483)
    bmm_397: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1483, permute_1853);  view_1483 = permute_1853 = None
    view_1484: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_396, [1024, 1, 16, 64, 1]);  bmm_396 = None
    permute_1854: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1484, [4, 1, 2, 3, 0]);  view_1484 = None
    view_1485: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_397, [512, 1024, 1, 1, 1]);  bmm_397 = None
    permute_1855: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1485, [0, 2, 3, 4, 1]);  view_1485 = None
    permute_1856: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1854, [4, 2, 3, 0, 1]);  permute_1854 = None
    squeeze_311: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1856, 4);  permute_1856 = None
    squeeze_312: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_311, 3);  squeeze_311 = None
    permute_1857: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1855, [0, 1, 4, 2, 3]);  permute_1855 = None
    squeeze_313: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1857, 4);  permute_1857 = None
    squeeze_314: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_313, 3);  squeeze_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_362: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_639, squeeze_314);  mul_639 = squeeze_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1486: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_307, [512, 1, 16, 64, 1]);  squeeze_307 = None
    permute_1858: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1486, [0, 4, 1, 2, 3]);  view_1486 = None
    view_1487: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1858, [1, 512, 1024]);  permute_1858 = None
    bmm_398: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1852, view_1487)
    bmm_399: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1487, permute_1860);  view_1487 = permute_1860 = None
    view_1488: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_398, [1024, 1, 16, 64, 1]);  bmm_398 = None
    permute_1861: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1488, [4, 1, 2, 3, 0]);  view_1488 = None
    view_1489: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_399, [512, 1024, 1, 1, 1]);  bmm_399 = None
    permute_1862: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1489, [0, 2, 3, 4, 1]);  view_1489 = None
    permute_1863: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1861, [4, 2, 3, 0, 1]);  permute_1861 = None
    squeeze_315: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1863, 4);  permute_1863 = None
    squeeze_316: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_315, 3);  squeeze_315 = None
    permute_1864: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1862, [0, 1, 4, 2, 3]);  permute_1862 = None
    squeeze_317: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1864, 4);  permute_1864 = None
    squeeze_318: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_317, 3);  squeeze_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_363: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_362, squeeze_318);  add_362 = squeeze_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1490: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_361, [512, 1, 16, 64, 1]);  add_361 = None
    permute_1865: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1490, [0, 4, 1, 2, 3]);  view_1490 = None
    clone_132: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1865, memory_format = torch.contiguous_format);  permute_1865 = None
    view_1491: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_132, [1, 512, 1024]);  clone_132 = None
    bmm_400: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1852, view_1491);  permute_1852 = None
    bmm_401: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1491, permute_1867);  view_1491 = permute_1867 = None
    view_1492: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_400, [1024, 1, 16, 64, 1]);  bmm_400 = None
    permute_1868: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1492, [4, 1, 2, 3, 0]);  view_1492 = None
    view_1493: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_401, [512, 1024, 1, 1, 1]);  bmm_401 = None
    permute_1869: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1493, [0, 2, 3, 4, 1]);  view_1493 = None
    permute_1870: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1868, [4, 2, 3, 0, 1]);  permute_1868 = None
    squeeze_319: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1870, 4);  permute_1870 = None
    squeeze_320: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_319, 3);  squeeze_319 = None
    permute_1871: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1869, [0, 1, 4, 2, 3]);  permute_1869 = None
    squeeze_321: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1871, 4);  permute_1871 = None
    squeeze_322: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_321, 3);  squeeze_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_364: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_363, squeeze_322);  add_363 = squeeze_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_649: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_364, primals_248);  primals_248 = None
    mul_650: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_649, 1024)
    sum_212: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_649, [2], True)
    mul_651: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_649, mul_82);  mul_649 = None
    sum_213: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_651, [2], True);  mul_651 = None
    mul_652: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_82, sum_213);  sum_213 = None
    sub_174: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_650, sum_212);  mul_650 = sum_212 = None
    sub_175: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_174, mul_652);  sub_174 = mul_652 = None
    mul_653: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_55, sub_175);  div_55 = sub_175 = None
    mul_654: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_364, mul_82);  mul_82 = None
    sum_214: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_654, [0, 1]);  mul_654 = None
    sum_215: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_364, [0, 1]);  add_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_62: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_655: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 1.1111111111111112);  convert_element_type_62 = None
    mul_656: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_653, mul_655);  mul_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1494: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_656, [512, 1024]);  mul_656 = None
    mm_58: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1494, permute_1872);  permute_1872 = None
    permute_1873: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1494, [1, 0])
    mm_59: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1873, view_378);  permute_1873 = view_378 = None
    permute_1874: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_216: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1494, [0], True);  view_1494 = None
    view_1495: "f32[1024]" = torch.ops.aten.reshape.default(sum_216, [1024]);  sum_216 = None
    permute_1875: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1874, [1, 0]);  permute_1874 = None
    view_1496: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_58, [512, 1, 4096]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_63: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_119, torch.float32);  getitem_119 = None
    mul_657: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 1.1111111111111112);  convert_element_type_63 = None
    mul_658: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1496, mul_657);  view_1496 = mul_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_660: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_108, 0.5);  add_108 = None
    mul_661: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_377, view_377)
    mul_662: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_661, -0.5);  mul_661 = None
    exp_40: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_662);  mul_662 = None
    mul_663: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
    mul_664: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_377, mul_663);  view_377 = mul_663 = None
    add_366: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_660, mul_664);  mul_660 = mul_664 = None
    mul_665: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_658, add_366);  mul_658 = add_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1497: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_665, [512, 4096]);  mul_665 = None
    mm_60: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1497, permute_1876);  permute_1876 = None
    permute_1877: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1497, [1, 0])
    mm_61: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1877, view_376);  permute_1877 = view_376 = None
    permute_1878: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_217: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1497, [0], True);  view_1497 = None
    view_1498: "f32[4096]" = torch.ops.aten.reshape.default(sum_217, [4096]);  sum_217 = None
    permute_1879: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1878, [1, 0]);  permute_1878 = None
    view_1499: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_60, [512, 1, 1024]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_367: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_653, view_1499);  mul_653 = view_1499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_667: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_367, primals_242);  primals_242 = None
    mul_668: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_667, 1024)
    sum_218: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_667, [2], True)
    mul_669: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_667, mul_77);  mul_667 = None
    sum_219: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_669, [2], True);  mul_669 = None
    mul_670: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_77, sum_219);  sum_219 = None
    sub_177: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_668, sum_218);  mul_668 = sum_218 = None
    sub_178: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_177, mul_670);  sub_177 = mul_670 = None
    mul_671: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_56, sub_178);  div_56 = sub_178 = None
    mul_672: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_367, mul_77);  mul_77 = None
    sum_220: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_672, [0, 1]);  mul_672 = None
    sum_221: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_367, [0, 1]);  add_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_64: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_115, torch.float32);  getitem_115 = None
    mul_673: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_64, 1.1111111111111112);  convert_element_type_64 = None
    mul_674: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_671, mul_673);  mul_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1500: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_674, [512, 1, 1024, 1, 1]);  mul_674 = None
    permute_1880: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1500, [0, 3, 4, 1, 2]);  view_1500 = None
    view_1501: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1880, [1, 512, 1024]);  permute_1880 = None
    bmm_402: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1881, view_1501);  permute_1881 = None
    bmm_403: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1501, permute_1882);  view_1501 = permute_1882 = None
    view_1502: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_402, [64, 16, 1, 1024, 1]);  bmm_402 = None
    permute_1883: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1502, [4, 2, 3, 0, 1]);  view_1502 = None
    view_1503: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_403, [512, 64, 16, 1, 1]);  bmm_403 = None
    permute_1884: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1503, [0, 3, 4, 1, 2]);  view_1503 = None
    permute_1885: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1883, [2, 4, 3, 0, 1]);  permute_1883 = None
    squeeze_323: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1885, 4);  permute_1885 = None
    squeeze_324: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_323, 3);  squeeze_323 = None
    permute_1886: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1884, [0, 1, 4, 3, 2]);  permute_1884 = None
    squeeze_325: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1886, 4);  permute_1886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1504: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_325, [512, 1, 16, 64, 1]);  squeeze_325 = None
    permute_1887: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1504, [2, 0, 4, 1, 3]);  view_1504 = None
    view_1505: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_1887, [16, 512, 64]);  permute_1887 = None
    bmm_404: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_1888, view_1505);  permute_1888 = None
    bmm_405: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1505, permute_1889);  view_1505 = permute_1889 = None
    view_1506: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_404, [16, 512, 1, 64, 1]);  bmm_404 = None
    permute_1890: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1506, [4, 2, 0, 3, 1]);  view_1506 = None
    view_1507: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_405, [16, 512, 512, 1, 1]);  bmm_405 = None
    permute_1891: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1507, [1, 3, 0, 4, 2]);  view_1507 = None
    permute_1892: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1890, [4, 1, 2, 3, 0]);  permute_1890 = None
    squeeze_326: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1892, 4);  permute_1892 = None
    permute_1893: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_1891, [1, 2, 0, 4, 3]);  permute_1891 = None
    squeeze_327: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_1893, 4);  permute_1893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_65: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_113, torch.float32);  getitem_113 = None
    mul_675: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_65, 1.1111111111111112);  convert_element_type_65 = None
    mul_676: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_327, mul_675);  squeeze_327 = mul_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_677: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_676, alias_40);  mul_676 = None
    sum_222: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_677, [3], True)
    mul_678: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_40, sum_222);  alias_40 = sum_222 = None
    sub_179: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_677, mul_678);  mul_677 = mul_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_679: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_179, 0.125);  sub_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_14: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_679, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1508: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_14, [1, 16, 1023, 512]);  index_put_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_57: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1508, 2, 1, 9223372036854775807);  view_1508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1509: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_57, [1, 16, 512, 1024]);  slice_scatter_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1510: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1509, [1, 16, 512, 1024, 1]);  view_1509 = None
    permute_1894: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1510, [1, 2, 4, 0, 3]);  view_1510 = None
    view_1511: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_1894, [16, 512, 1024]);  permute_1894 = None
    bmm_406: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_1895, view_1511);  permute_1895 = None
    bmm_407: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1511, permute_1896);  view_1511 = permute_1896 = None
    view_1512: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_406, [16, 64, 1, 1024, 1]);  bmm_406 = None
    permute_1897: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1512, [2, 0, 4, 3, 1]);  view_1512 = None
    view_1513: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_407, [16, 512, 64, 1, 1]);  bmm_407 = None
    permute_1898: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1513, [3, 0, 1, 4, 2]);  view_1513 = None
    permute_1899: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1897, [3, 0, 1, 4, 2]);  permute_1897 = None
    squeeze_328: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1899, 4);  permute_1899 = None
    permute_1900: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1898, [2, 0, 1, 4, 3]);  permute_1898 = None
    squeeze_329: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1900, 4);  permute_1900 = None
    sum_223: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_329, [0, 1], True)
    view_1514: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_223, [16, 64]);  sum_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1515: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_679, [1, 16, 512, 512, 1]);  mul_679 = None
    permute_1901: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1515, [1, 2, 4, 0, 3]);  view_1515 = None
    view_1516: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_1901, [16, 512, 512]);  permute_1901 = None
    bmm_408: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_1902, view_1516);  permute_1902 = None
    bmm_409: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1516, permute_1903);  view_1516 = permute_1903 = None
    view_1517: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_408, [16, 64, 1, 512, 1]);  bmm_408 = None
    permute_1904: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1517, [2, 0, 4, 3, 1]);  view_1517 = None
    view_1518: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_409, [16, 512, 64, 1, 1]);  bmm_409 = None
    permute_1905: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1518, [3, 0, 1, 4, 2]);  view_1518 = None
    permute_1906: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1904, [3, 0, 1, 4, 2]);  permute_1904 = None
    squeeze_330: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1906, 4);  permute_1906 = None
    permute_1907: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1905, [2, 0, 1, 4, 3]);  permute_1905 = None
    squeeze_331: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1907, 4);  permute_1907 = None
    sum_224: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_331, [0, 1], True)
    view_1519: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_224, [16, 64]);  sum_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_368: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_329, squeeze_331);  squeeze_329 = squeeze_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1520: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_328, [1024, 1, 16, 64, 1]);  squeeze_328 = None
    permute_1908: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1520, [0, 4, 1, 2, 3]);  view_1520 = None
    view_1521: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_1908, [1, 1024, 1024]);  permute_1908 = None
    bmm_410: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1521);  view_1521 = None
    view_1522: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_410, [1024, 1, 16, 64, 1]);  bmm_410 = None
    permute_1910: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1522, [4, 1, 2, 3, 0]);  view_1522 = None
    permute_1911: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1910, [4, 2, 3, 0, 1]);  permute_1910 = None
    squeeze_332: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1911, 4);  permute_1911 = None
    squeeze_333: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_332, 3);  squeeze_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1523: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_326, [512, 1, 16, 64, 1]);  squeeze_326 = None
    permute_1912: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1523, [0, 4, 1, 2, 3]);  view_1523 = None
    clone_137: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1912, memory_format = torch.contiguous_format);  permute_1912 = None
    view_1524: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_137, [1, 512, 1024]);  clone_137 = None
    bmm_411: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1913, view_1524)
    bmm_412: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1524, permute_1914);  view_1524 = permute_1914 = None
    view_1525: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_411, [1024, 1, 16, 64, 1]);  bmm_411 = None
    permute_1915: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1525, [4, 1, 2, 3, 0]);  view_1525 = None
    view_1526: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_412, [512, 1024, 1, 1, 1]);  bmm_412 = None
    permute_1916: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1526, [0, 2, 3, 4, 1]);  view_1526 = None
    permute_1917: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1915, [4, 2, 3, 0, 1]);  permute_1915 = None
    squeeze_334: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1917, 4);  permute_1917 = None
    squeeze_335: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_334, 3);  squeeze_334 = None
    permute_1918: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1916, [0, 1, 4, 2, 3]);  permute_1916 = None
    squeeze_336: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1918, 4);  permute_1918 = None
    squeeze_337: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_336, 3);  squeeze_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_369: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_671, squeeze_337);  mul_671 = squeeze_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1527: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_330, [512, 1, 16, 64, 1]);  squeeze_330 = None
    permute_1919: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1527, [0, 4, 1, 2, 3]);  view_1527 = None
    view_1528: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1919, [1, 512, 1024]);  permute_1919 = None
    bmm_413: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1913, view_1528)
    bmm_414: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1528, permute_1921);  view_1528 = permute_1921 = None
    view_1529: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_413, [1024, 1, 16, 64, 1]);  bmm_413 = None
    permute_1922: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1529, [4, 1, 2, 3, 0]);  view_1529 = None
    view_1530: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_414, [512, 1024, 1, 1, 1]);  bmm_414 = None
    permute_1923: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1530, [0, 2, 3, 4, 1]);  view_1530 = None
    permute_1924: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1922, [4, 2, 3, 0, 1]);  permute_1922 = None
    squeeze_338: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1924, 4);  permute_1924 = None
    squeeze_339: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_338, 3);  squeeze_338 = None
    permute_1925: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1923, [0, 1, 4, 2, 3]);  permute_1923 = None
    squeeze_340: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1925, 4);  permute_1925 = None
    squeeze_341: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_340, 3);  squeeze_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_370: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_369, squeeze_341);  add_369 = squeeze_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1531: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_368, [512, 1, 16, 64, 1]);  add_368 = None
    permute_1926: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1531, [0, 4, 1, 2, 3]);  view_1531 = None
    clone_138: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1926, memory_format = torch.contiguous_format);  permute_1926 = None
    view_1532: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_138, [1, 512, 1024]);  clone_138 = None
    bmm_415: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1913, view_1532);  permute_1913 = None
    bmm_416: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1532, permute_1928);  view_1532 = permute_1928 = None
    view_1533: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_415, [1024, 1, 16, 64, 1]);  bmm_415 = None
    permute_1929: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1533, [4, 1, 2, 3, 0]);  view_1533 = None
    view_1534: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_416, [512, 1024, 1, 1, 1]);  bmm_416 = None
    permute_1930: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1534, [0, 2, 3, 4, 1]);  view_1534 = None
    permute_1931: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1929, [4, 2, 3, 0, 1]);  permute_1929 = None
    squeeze_342: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1931, 4);  permute_1931 = None
    squeeze_343: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_342, 3);  squeeze_342 = None
    permute_1932: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1930, [0, 1, 4, 2, 3]);  permute_1930 = None
    squeeze_344: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1932, 4);  permute_1932 = None
    squeeze_345: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_344, 3);  squeeze_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_371: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_370, squeeze_345);  add_370 = squeeze_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_681: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_371, primals_240);  primals_240 = None
    mul_682: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_681, 1024)
    sum_225: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_681, [2], True)
    mul_683: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_681, mul_74);  mul_681 = None
    sum_226: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_683, [2], True);  mul_683 = None
    mul_684: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_74, sum_226);  sum_226 = None
    sub_181: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_682, sum_225);  mul_682 = sum_225 = None
    sub_182: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_181, mul_684);  sub_181 = mul_684 = None
    mul_685: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_57, sub_182);  div_57 = sub_182 = None
    mul_686: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_371, mul_74);  mul_74 = None
    sum_227: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_686, [0, 1]);  mul_686 = None
    sum_228: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_371, [0, 1]);  add_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_66: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_109, torch.float32);  getitem_109 = None
    mul_687: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
    mul_688: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_685, mul_687);  mul_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1535: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_688, [512, 1024]);  mul_688 = None
    mm_62: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1535, permute_1933);  permute_1933 = None
    permute_1934: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1535, [1, 0])
    mm_63: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1934, view_340);  permute_1934 = view_340 = None
    permute_1935: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_229: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1535, [0], True);  view_1535 = None
    view_1536: "f32[1024]" = torch.ops.aten.reshape.default(sum_229, [1024]);  sum_229 = None
    permute_1936: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1935, [1, 0]);  permute_1935 = None
    view_1537: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_62, [512, 1, 4096]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_67: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_689: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 1.1111111111111112);  convert_element_type_67 = None
    mul_690: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1537, mul_689);  view_1537 = mul_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_692: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_97, 0.5);  add_97 = None
    mul_693: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_339, view_339)
    mul_694: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_693, -0.5);  mul_693 = None
    exp_41: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_694);  mul_694 = None
    mul_695: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
    mul_696: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_339, mul_695);  view_339 = mul_695 = None
    add_373: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_692, mul_696);  mul_692 = mul_696 = None
    mul_697: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_690, add_373);  mul_690 = add_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1538: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_697, [512, 4096]);  mul_697 = None
    mm_64: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1538, permute_1937);  permute_1937 = None
    permute_1938: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1538, [1, 0])
    mm_65: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1938, view_338);  permute_1938 = view_338 = None
    permute_1939: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_230: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1538, [0], True);  view_1538 = None
    view_1539: "f32[4096]" = torch.ops.aten.reshape.default(sum_230, [4096]);  sum_230 = None
    permute_1940: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1939, [1, 0]);  permute_1939 = None
    view_1540: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_64, [512, 1, 1024]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_374: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_685, view_1540);  mul_685 = view_1540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_699: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_374, primals_234);  primals_234 = None
    mul_700: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_699, 1024)
    sum_231: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_699, [2], True)
    mul_701: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_699, mul_69);  mul_699 = None
    sum_232: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_701, [2], True);  mul_701 = None
    mul_702: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_69, sum_232);  sum_232 = None
    sub_184: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_700, sum_231);  mul_700 = sum_231 = None
    sub_185: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_184, mul_702);  sub_184 = mul_702 = None
    mul_703: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_58, sub_185);  div_58 = sub_185 = None
    mul_704: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_374, mul_69);  mul_69 = None
    sum_233: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_704, [0, 1]);  mul_704 = None
    sum_234: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_374, [0, 1]);  add_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_68: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_103, torch.float32);  getitem_103 = None
    mul_705: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
    mul_706: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_703, mul_705);  mul_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1541: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_706, [512, 1, 1024, 1, 1]);  mul_706 = None
    permute_1941: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1541, [0, 3, 4, 1, 2]);  view_1541 = None
    view_1542: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1941, [1, 512, 1024]);  permute_1941 = None
    bmm_417: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1942, view_1542);  permute_1942 = None
    bmm_418: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1542, permute_1943);  view_1542 = permute_1943 = None
    view_1543: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_417, [64, 16, 1, 1024, 1]);  bmm_417 = None
    permute_1944: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1543, [4, 2, 3, 0, 1]);  view_1543 = None
    view_1544: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_418, [512, 64, 16, 1, 1]);  bmm_418 = None
    permute_1945: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1544, [0, 3, 4, 1, 2]);  view_1544 = None
    permute_1946: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1944, [2, 4, 3, 0, 1]);  permute_1944 = None
    squeeze_346: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1946, 4);  permute_1946 = None
    squeeze_347: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_346, 3);  squeeze_346 = None
    permute_1947: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1945, [0, 1, 4, 3, 2]);  permute_1945 = None
    squeeze_348: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1947, 4);  permute_1947 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1545: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_348, [512, 1, 16, 64, 1]);  squeeze_348 = None
    permute_1948: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1545, [2, 0, 4, 1, 3]);  view_1545 = None
    view_1546: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_1948, [16, 512, 64]);  permute_1948 = None
    bmm_419: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_1949, view_1546);  permute_1949 = None
    bmm_420: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1546, permute_1950);  view_1546 = permute_1950 = None
    view_1547: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_419, [16, 512, 1, 64, 1]);  bmm_419 = None
    permute_1951: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1547, [4, 2, 0, 3, 1]);  view_1547 = None
    view_1548: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_420, [16, 512, 512, 1, 1]);  bmm_420 = None
    permute_1952: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1548, [1, 3, 0, 4, 2]);  view_1548 = None
    permute_1953: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1951, [4, 1, 2, 3, 0]);  permute_1951 = None
    squeeze_349: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1953, 4);  permute_1953 = None
    permute_1954: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_1952, [1, 2, 0, 4, 3]);  permute_1952 = None
    squeeze_350: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_1954, 4);  permute_1954 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_69: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_707: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_69, 1.1111111111111112);  convert_element_type_69 = None
    mul_708: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_350, mul_707);  squeeze_350 = mul_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_709: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_708, alias_41);  mul_708 = None
    sum_235: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_709, [3], True)
    mul_710: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_41, sum_235);  alias_41 = sum_235 = None
    sub_186: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_709, mul_710);  mul_709 = mul_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_711: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_186, 0.125);  sub_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_15: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_711, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1549: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_15, [1, 16, 1023, 512]);  index_put_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_61: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1549, 2, 1, 9223372036854775807);  view_1549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1550: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_61, [1, 16, 512, 1024]);  slice_scatter_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1551: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1550, [1, 16, 512, 1024, 1]);  view_1550 = None
    permute_1955: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1551, [1, 2, 4, 0, 3]);  view_1551 = None
    view_1552: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_1955, [16, 512, 1024]);  permute_1955 = None
    bmm_421: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_1956, view_1552);  permute_1956 = None
    bmm_422: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1552, permute_1957);  view_1552 = permute_1957 = None
    view_1553: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_421, [16, 64, 1, 1024, 1]);  bmm_421 = None
    permute_1958: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1553, [2, 0, 4, 3, 1]);  view_1553 = None
    view_1554: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_422, [16, 512, 64, 1, 1]);  bmm_422 = None
    permute_1959: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1554, [3, 0, 1, 4, 2]);  view_1554 = None
    permute_1960: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1958, [3, 0, 1, 4, 2]);  permute_1958 = None
    squeeze_351: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1960, 4);  permute_1960 = None
    permute_1961: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1959, [2, 0, 1, 4, 3]);  permute_1959 = None
    squeeze_352: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1961, 4);  permute_1961 = None
    sum_236: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_352, [0, 1], True)
    view_1555: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_236, [16, 64]);  sum_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1556: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_711, [1, 16, 512, 512, 1]);  mul_711 = None
    permute_1962: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1556, [1, 2, 4, 0, 3]);  view_1556 = None
    view_1557: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_1962, [16, 512, 512]);  permute_1962 = None
    bmm_423: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_1963, view_1557);  permute_1963 = None
    bmm_424: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1557, permute_1964);  view_1557 = permute_1964 = None
    view_1558: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_423, [16, 64, 1, 512, 1]);  bmm_423 = None
    permute_1965: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1558, [2, 0, 4, 3, 1]);  view_1558 = None
    view_1559: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_424, [16, 512, 64, 1, 1]);  bmm_424 = None
    permute_1966: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1559, [3, 0, 1, 4, 2]);  view_1559 = None
    permute_1967: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1965, [3, 0, 1, 4, 2]);  permute_1965 = None
    squeeze_353: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1967, 4);  permute_1967 = None
    permute_1968: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1966, [2, 0, 1, 4, 3]);  permute_1966 = None
    squeeze_354: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1968, 4);  permute_1968 = None
    sum_237: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_354, [0, 1], True)
    view_1560: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_237, [16, 64]);  sum_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_375: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_352, squeeze_354);  squeeze_352 = squeeze_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1561: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_351, [1024, 1, 16, 64, 1]);  squeeze_351 = None
    permute_1969: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1561, [0, 4, 1, 2, 3]);  view_1561 = None
    view_1562: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_1969, [1, 1024, 1024]);  permute_1969 = None
    bmm_425: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1562);  view_1562 = None
    view_1563: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_425, [1024, 1, 16, 64, 1]);  bmm_425 = None
    permute_1971: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1563, [4, 1, 2, 3, 0]);  view_1563 = None
    permute_1972: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1971, [4, 2, 3, 0, 1]);  permute_1971 = None
    squeeze_355: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1972, 4);  permute_1972 = None
    squeeze_356: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_355, 3);  squeeze_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1564: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_349, [512, 1, 16, 64, 1]);  squeeze_349 = None
    permute_1973: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1564, [0, 4, 1, 2, 3]);  view_1564 = None
    clone_143: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1973, memory_format = torch.contiguous_format);  permute_1973 = None
    view_1565: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_143, [1, 512, 1024]);  clone_143 = None
    bmm_426: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1974, view_1565)
    bmm_427: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1565, permute_1975);  view_1565 = permute_1975 = None
    view_1566: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_426, [1024, 1, 16, 64, 1]);  bmm_426 = None
    permute_1976: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1566, [4, 1, 2, 3, 0]);  view_1566 = None
    view_1567: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_427, [512, 1024, 1, 1, 1]);  bmm_427 = None
    permute_1977: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1567, [0, 2, 3, 4, 1]);  view_1567 = None
    permute_1978: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1976, [4, 2, 3, 0, 1]);  permute_1976 = None
    squeeze_357: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1978, 4);  permute_1978 = None
    squeeze_358: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_357, 3);  squeeze_357 = None
    permute_1979: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1977, [0, 1, 4, 2, 3]);  permute_1977 = None
    squeeze_359: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1979, 4);  permute_1979 = None
    squeeze_360: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_359, 3);  squeeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_376: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_703, squeeze_360);  mul_703 = squeeze_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1568: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_353, [512, 1, 16, 64, 1]);  squeeze_353 = None
    permute_1980: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1568, [0, 4, 1, 2, 3]);  view_1568 = None
    view_1569: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_1980, [1, 512, 1024]);  permute_1980 = None
    bmm_428: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1974, view_1569)
    bmm_429: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1569, permute_1982);  view_1569 = permute_1982 = None
    view_1570: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_428, [1024, 1, 16, 64, 1]);  bmm_428 = None
    permute_1983: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1570, [4, 1, 2, 3, 0]);  view_1570 = None
    view_1571: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_429, [512, 1024, 1, 1, 1]);  bmm_429 = None
    permute_1984: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1571, [0, 2, 3, 4, 1]);  view_1571 = None
    permute_1985: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1983, [4, 2, 3, 0, 1]);  permute_1983 = None
    squeeze_361: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1985, 4);  permute_1985 = None
    squeeze_362: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_361, 3);  squeeze_361 = None
    permute_1986: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1984, [0, 1, 4, 2, 3]);  permute_1984 = None
    squeeze_363: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1986, 4);  permute_1986 = None
    squeeze_364: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_363, 3);  squeeze_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_377: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_376, squeeze_364);  add_376 = squeeze_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1572: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_375, [512, 1, 16, 64, 1]);  add_375 = None
    permute_1987: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1572, [0, 4, 1, 2, 3]);  view_1572 = None
    clone_144: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_1987, memory_format = torch.contiguous_format);  permute_1987 = None
    view_1573: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_144, [1, 512, 1024]);  clone_144 = None
    bmm_430: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1974, view_1573);  permute_1974 = None
    bmm_431: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1573, permute_1989);  view_1573 = permute_1989 = None
    view_1574: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_430, [1024, 1, 16, 64, 1]);  bmm_430 = None
    permute_1990: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1574, [4, 1, 2, 3, 0]);  view_1574 = None
    view_1575: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_431, [512, 1024, 1, 1, 1]);  bmm_431 = None
    permute_1991: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1575, [0, 2, 3, 4, 1]);  view_1575 = None
    permute_1992: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_1990, [4, 2, 3, 0, 1]);  permute_1990 = None
    squeeze_365: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_1992, 4);  permute_1992 = None
    squeeze_366: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_365, 3);  squeeze_365 = None
    permute_1993: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_1991, [0, 1, 4, 2, 3]);  permute_1991 = None
    squeeze_367: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_1993, 4);  permute_1993 = None
    squeeze_368: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_367, 3);  squeeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_378: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_377, squeeze_368);  add_377 = squeeze_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_713: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_378, primals_232);  primals_232 = None
    mul_714: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_713, 1024)
    sum_238: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_713, [2], True)
    mul_715: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_713, mul_66);  mul_713 = None
    sum_239: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_715, [2], True);  mul_715 = None
    mul_716: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_66, sum_239);  sum_239 = None
    sub_188: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_714, sum_238);  mul_714 = sum_238 = None
    sub_189: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_188, mul_716);  sub_188 = mul_716 = None
    mul_717: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_59, sub_189);  div_59 = sub_189 = None
    mul_718: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_378, mul_66);  mul_66 = None
    sum_240: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_718, [0, 1]);  mul_718 = None
    sum_241: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_378, [0, 1]);  add_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_70: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_719: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 1.1111111111111112);  convert_element_type_70 = None
    mul_720: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_717, mul_719);  mul_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1576: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_720, [512, 1024]);  mul_720 = None
    mm_66: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1576, permute_1994);  permute_1994 = None
    permute_1995: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1576, [1, 0])
    mm_67: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1995, view_302);  permute_1995 = view_302 = None
    permute_1996: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_242: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1576, [0], True);  view_1576 = None
    view_1577: "f32[1024]" = torch.ops.aten.reshape.default(sum_242, [1024]);  sum_242 = None
    permute_1997: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1996, [1, 0]);  permute_1996 = None
    view_1578: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_66, [512, 1, 4096]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_71: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_95, torch.float32);  getitem_95 = None
    mul_721: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_71, 1.1111111111111112);  convert_element_type_71 = None
    mul_722: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1578, mul_721);  view_1578 = mul_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_724: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_86, 0.5);  add_86 = None
    mul_725: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_301, view_301)
    mul_726: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_725, -0.5);  mul_725 = None
    exp_42: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_726);  mul_726 = None
    mul_727: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
    mul_728: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_301, mul_727);  view_301 = mul_727 = None
    add_380: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_724, mul_728);  mul_724 = mul_728 = None
    mul_729: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_722, add_380);  mul_722 = add_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1579: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_729, [512, 4096]);  mul_729 = None
    mm_68: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1579, permute_1998);  permute_1998 = None
    permute_1999: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1579, [1, 0])
    mm_69: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1999, view_300);  permute_1999 = view_300 = None
    permute_2000: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_243: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1579, [0], True);  view_1579 = None
    view_1580: "f32[4096]" = torch.ops.aten.reshape.default(sum_243, [4096]);  sum_243 = None
    permute_2001: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_2000, [1, 0]);  permute_2000 = None
    view_1581: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_68, [512, 1, 1024]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_381: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_717, view_1581);  mul_717 = view_1581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_731: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_381, primals_226);  primals_226 = None
    mul_732: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_731, 1024)
    sum_244: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_731, [2], True)
    mul_733: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_731, mul_61);  mul_731 = None
    sum_245: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_733, [2], True);  mul_733 = None
    mul_734: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_61, sum_245);  sum_245 = None
    sub_191: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_732, sum_244);  mul_732 = sum_244 = None
    sub_192: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_191, mul_734);  sub_191 = mul_734 = None
    mul_735: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_60, sub_192);  div_60 = sub_192 = None
    mul_736: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_381, mul_61);  mul_61 = None
    sum_246: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_736, [0, 1]);  mul_736 = None
    sum_247: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_381, [0, 1]);  add_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_72: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_737: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_72, 1.1111111111111112);  convert_element_type_72 = None
    mul_738: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_735, mul_737);  mul_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1582: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_738, [512, 1, 1024, 1, 1]);  mul_738 = None
    permute_2002: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1582, [0, 3, 4, 1, 2]);  view_1582 = None
    view_1583: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_2002, [1, 512, 1024]);  permute_2002 = None
    bmm_432: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2003, view_1583);  permute_2003 = None
    bmm_433: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1583, permute_2004);  view_1583 = permute_2004 = None
    view_1584: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_432, [64, 16, 1, 1024, 1]);  bmm_432 = None
    permute_2005: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1584, [4, 2, 3, 0, 1]);  view_1584 = None
    view_1585: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_433, [512, 64, 16, 1, 1]);  bmm_433 = None
    permute_2006: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1585, [0, 3, 4, 1, 2]);  view_1585 = None
    permute_2007: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2005, [2, 4, 3, 0, 1]);  permute_2005 = None
    squeeze_369: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2007, 4);  permute_2007 = None
    squeeze_370: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_369, 3);  squeeze_369 = None
    permute_2008: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2006, [0, 1, 4, 3, 2]);  permute_2006 = None
    squeeze_371: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2008, 4);  permute_2008 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1586: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_371, [512, 1, 16, 64, 1]);  squeeze_371 = None
    permute_2009: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1586, [2, 0, 4, 1, 3]);  view_1586 = None
    view_1587: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_2009, [16, 512, 64]);  permute_2009 = None
    bmm_434: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_2010, view_1587);  permute_2010 = None
    bmm_435: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1587, permute_2011);  view_1587 = permute_2011 = None
    view_1588: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_434, [16, 512, 1, 64, 1]);  bmm_434 = None
    permute_2012: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1588, [4, 2, 0, 3, 1]);  view_1588 = None
    view_1589: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_435, [16, 512, 512, 1, 1]);  bmm_435 = None
    permute_2013: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1589, [1, 3, 0, 4, 2]);  view_1589 = None
    permute_2014: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2012, [4, 1, 2, 3, 0]);  permute_2012 = None
    squeeze_372: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2014, 4);  permute_2014 = None
    permute_2015: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_2013, [1, 2, 0, 4, 3]);  permute_2013 = None
    squeeze_373: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_2015, 4);  permute_2015 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_73: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_89, torch.float32);  getitem_89 = None
    mul_739: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_73, 1.1111111111111112);  convert_element_type_73 = None
    mul_740: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_373, mul_739);  squeeze_373 = mul_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_741: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_740, alias_42);  mul_740 = None
    sum_248: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_741, [3], True)
    mul_742: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_42, sum_248);  alias_42 = sum_248 = None
    sub_193: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_741, mul_742);  mul_741 = mul_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_743: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_193, 0.125);  sub_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_16: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_743, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1590: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_16, [1, 16, 1023, 512]);  index_put_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_65: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1590, 2, 1, 9223372036854775807);  view_1590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1591: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_65, [1, 16, 512, 1024]);  slice_scatter_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1592: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1591, [1, 16, 512, 1024, 1]);  view_1591 = None
    permute_2016: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1592, [1, 2, 4, 0, 3]);  view_1592 = None
    view_1593: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_2016, [16, 512, 1024]);  permute_2016 = None
    bmm_436: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_2017, view_1593);  permute_2017 = None
    bmm_437: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1593, permute_2018);  view_1593 = permute_2018 = None
    view_1594: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_436, [16, 64, 1, 1024, 1]);  bmm_436 = None
    permute_2019: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1594, [2, 0, 4, 3, 1]);  view_1594 = None
    view_1595: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_437, [16, 512, 64, 1, 1]);  bmm_437 = None
    permute_2020: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1595, [3, 0, 1, 4, 2]);  view_1595 = None
    permute_2021: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2019, [3, 0, 1, 4, 2]);  permute_2019 = None
    squeeze_374: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2021, 4);  permute_2021 = None
    permute_2022: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2020, [2, 0, 1, 4, 3]);  permute_2020 = None
    squeeze_375: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2022, 4);  permute_2022 = None
    sum_249: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_375, [0, 1], True)
    view_1596: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_249, [16, 64]);  sum_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1597: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_743, [1, 16, 512, 512, 1]);  mul_743 = None
    permute_2023: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1597, [1, 2, 4, 0, 3]);  view_1597 = None
    view_1598: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_2023, [16, 512, 512]);  permute_2023 = None
    bmm_438: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_2024, view_1598);  permute_2024 = None
    bmm_439: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1598, permute_2025);  view_1598 = permute_2025 = None
    view_1599: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_438, [16, 64, 1, 512, 1]);  bmm_438 = None
    permute_2026: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1599, [2, 0, 4, 3, 1]);  view_1599 = None
    view_1600: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_439, [16, 512, 64, 1, 1]);  bmm_439 = None
    permute_2027: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1600, [3, 0, 1, 4, 2]);  view_1600 = None
    permute_2028: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2026, [3, 0, 1, 4, 2]);  permute_2026 = None
    squeeze_376: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2028, 4);  permute_2028 = None
    permute_2029: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2027, [2, 0, 1, 4, 3]);  permute_2027 = None
    squeeze_377: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2029, 4);  permute_2029 = None
    sum_250: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_377, [0, 1], True)
    view_1601: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_250, [16, 64]);  sum_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_382: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_375, squeeze_377);  squeeze_375 = squeeze_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1602: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_374, [1024, 1, 16, 64, 1]);  squeeze_374 = None
    permute_2030: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1602, [0, 4, 1, 2, 3]);  view_1602 = None
    view_1603: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_2030, [1, 1024, 1024]);  permute_2030 = None
    bmm_440: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1603);  view_1603 = None
    view_1604: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_440, [1024, 1, 16, 64, 1]);  bmm_440 = None
    permute_2032: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1604, [4, 1, 2, 3, 0]);  view_1604 = None
    permute_2033: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2032, [4, 2, 3, 0, 1]);  permute_2032 = None
    squeeze_378: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2033, 4);  permute_2033 = None
    squeeze_379: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_378, 3);  squeeze_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1605: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_372, [512, 1, 16, 64, 1]);  squeeze_372 = None
    permute_2034: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1605, [0, 4, 1, 2, 3]);  view_1605 = None
    clone_149: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_2034, memory_format = torch.contiguous_format);  permute_2034 = None
    view_1606: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_149, [1, 512, 1024]);  clone_149 = None
    bmm_441: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2035, view_1606)
    bmm_442: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1606, permute_2036);  view_1606 = permute_2036 = None
    view_1607: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_441, [1024, 1, 16, 64, 1]);  bmm_441 = None
    permute_2037: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1607, [4, 1, 2, 3, 0]);  view_1607 = None
    view_1608: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_442, [512, 1024, 1, 1, 1]);  bmm_442 = None
    permute_2038: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1608, [0, 2, 3, 4, 1]);  view_1608 = None
    permute_2039: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2037, [4, 2, 3, 0, 1]);  permute_2037 = None
    squeeze_380: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2039, 4);  permute_2039 = None
    squeeze_381: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_380, 3);  squeeze_380 = None
    permute_2040: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2038, [0, 1, 4, 2, 3]);  permute_2038 = None
    squeeze_382: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2040, 4);  permute_2040 = None
    squeeze_383: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_382, 3);  squeeze_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_383: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_735, squeeze_383);  mul_735 = squeeze_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1609: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_376, [512, 1, 16, 64, 1]);  squeeze_376 = None
    permute_2041: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1609, [0, 4, 1, 2, 3]);  view_1609 = None
    view_1610: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_2041, [1, 512, 1024]);  permute_2041 = None
    bmm_443: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2035, view_1610)
    bmm_444: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1610, permute_2043);  view_1610 = permute_2043 = None
    view_1611: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_443, [1024, 1, 16, 64, 1]);  bmm_443 = None
    permute_2044: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1611, [4, 1, 2, 3, 0]);  view_1611 = None
    view_1612: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_444, [512, 1024, 1, 1, 1]);  bmm_444 = None
    permute_2045: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1612, [0, 2, 3, 4, 1]);  view_1612 = None
    permute_2046: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2044, [4, 2, 3, 0, 1]);  permute_2044 = None
    squeeze_384: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2046, 4);  permute_2046 = None
    squeeze_385: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_384, 3);  squeeze_384 = None
    permute_2047: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2045, [0, 1, 4, 2, 3]);  permute_2045 = None
    squeeze_386: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2047, 4);  permute_2047 = None
    squeeze_387: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_386, 3);  squeeze_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_384: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_383, squeeze_387);  add_383 = squeeze_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1613: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_382, [512, 1, 16, 64, 1]);  add_382 = None
    permute_2048: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1613, [0, 4, 1, 2, 3]);  view_1613 = None
    clone_150: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_2048, memory_format = torch.contiguous_format);  permute_2048 = None
    view_1614: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_150, [1, 512, 1024]);  clone_150 = None
    bmm_445: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2035, view_1614);  permute_2035 = None
    bmm_446: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1614, permute_2050);  view_1614 = permute_2050 = None
    view_1615: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_445, [1024, 1, 16, 64, 1]);  bmm_445 = None
    permute_2051: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1615, [4, 1, 2, 3, 0]);  view_1615 = None
    view_1616: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_446, [512, 1024, 1, 1, 1]);  bmm_446 = None
    permute_2052: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1616, [0, 2, 3, 4, 1]);  view_1616 = None
    permute_2053: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2051, [4, 2, 3, 0, 1]);  permute_2051 = None
    squeeze_388: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2053, 4);  permute_2053 = None
    squeeze_389: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_388, 3);  squeeze_388 = None
    permute_2054: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2052, [0, 1, 4, 2, 3]);  permute_2052 = None
    squeeze_390: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2054, 4);  permute_2054 = None
    squeeze_391: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_390, 3);  squeeze_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_385: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_384, squeeze_391);  add_384 = squeeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_745: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_385, primals_224);  primals_224 = None
    mul_746: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_745, 1024)
    sum_251: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_745, [2], True)
    mul_747: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_745, mul_58);  mul_745 = None
    sum_252: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_747, [2], True);  mul_747 = None
    mul_748: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_58, sum_252);  sum_252 = None
    sub_195: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_746, sum_251);  mul_746 = sum_251 = None
    sub_196: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_195, mul_748);  sub_195 = mul_748 = None
    mul_749: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_61, sub_196);  div_61 = sub_196 = None
    mul_750: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_385, mul_58);  mul_58 = None
    sum_253: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_750, [0, 1]);  mul_750 = None
    sum_254: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_385, [0, 1]);  add_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_74: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_85, torch.float32);  getitem_85 = None
    mul_751: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_74, 1.1111111111111112);  convert_element_type_74 = None
    mul_752: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_749, mul_751);  mul_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1617: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_752, [512, 1024]);  mul_752 = None
    mm_70: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1617, permute_2055);  permute_2055 = None
    permute_2056: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1617, [1, 0])
    mm_71: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_2056, view_264);  permute_2056 = view_264 = None
    permute_2057: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_255: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1617, [0], True);  view_1617 = None
    view_1618: "f32[1024]" = torch.ops.aten.reshape.default(sum_255, [1024]);  sum_255 = None
    permute_2058: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_2057, [1, 0]);  permute_2057 = None
    view_1619: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_70, [512, 1, 4096]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_75: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_83, torch.float32);  getitem_83 = None
    mul_753: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_75, 1.1111111111111112);  convert_element_type_75 = None
    mul_754: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1619, mul_753);  view_1619 = mul_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_756: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_75, 0.5);  add_75 = None
    mul_757: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_263, view_263)
    mul_758: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_757, -0.5);  mul_757 = None
    exp_43: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_758);  mul_758 = None
    mul_759: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
    mul_760: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_263, mul_759);  view_263 = mul_759 = None
    add_387: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_756, mul_760);  mul_756 = mul_760 = None
    mul_761: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_754, add_387);  mul_754 = add_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1620: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_761, [512, 4096]);  mul_761 = None
    mm_72: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1620, permute_2059);  permute_2059 = None
    permute_2060: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1620, [1, 0])
    mm_73: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_2060, view_262);  permute_2060 = view_262 = None
    permute_2061: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_256: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1620, [0], True);  view_1620 = None
    view_1621: "f32[4096]" = torch.ops.aten.reshape.default(sum_256, [4096]);  sum_256 = None
    permute_2062: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_2061, [1, 0]);  permute_2061 = None
    view_1622: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_72, [512, 1, 1024]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_388: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_749, view_1622);  mul_749 = view_1622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_763: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_388, primals_218);  primals_218 = None
    mul_764: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_763, 1024)
    sum_257: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_763, [2], True)
    mul_765: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_763, mul_53);  mul_763 = None
    sum_258: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_765, [2], True);  mul_765 = None
    mul_766: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_53, sum_258);  sum_258 = None
    sub_198: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_764, sum_257);  mul_764 = sum_257 = None
    sub_199: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_198, mul_766);  sub_198 = mul_766 = None
    mul_767: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_62, sub_199);  div_62 = sub_199 = None
    mul_768: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_388, mul_53);  mul_53 = None
    sum_259: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_768, [0, 1]);  mul_768 = None
    sum_260: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_388, [0, 1]);  add_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_76: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_79, torch.float32);  getitem_79 = None
    mul_769: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_76, 1.1111111111111112);  convert_element_type_76 = None
    mul_770: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_767, mul_769);  mul_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1623: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_770, [512, 1, 1024, 1, 1]);  mul_770 = None
    permute_2063: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1623, [0, 3, 4, 1, 2]);  view_1623 = None
    view_1624: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_2063, [1, 512, 1024]);  permute_2063 = None
    bmm_447: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2064, view_1624);  permute_2064 = None
    bmm_448: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1624, permute_2065);  view_1624 = permute_2065 = None
    view_1625: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_447, [64, 16, 1, 1024, 1]);  bmm_447 = None
    permute_2066: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1625, [4, 2, 3, 0, 1]);  view_1625 = None
    view_1626: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_448, [512, 64, 16, 1, 1]);  bmm_448 = None
    permute_2067: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1626, [0, 3, 4, 1, 2]);  view_1626 = None
    permute_2068: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2066, [2, 4, 3, 0, 1]);  permute_2066 = None
    squeeze_392: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2068, 4);  permute_2068 = None
    squeeze_393: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_392, 3);  squeeze_392 = None
    permute_2069: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2067, [0, 1, 4, 3, 2]);  permute_2067 = None
    squeeze_394: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2069, 4);  permute_2069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1627: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_394, [512, 1, 16, 64, 1]);  squeeze_394 = None
    permute_2070: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1627, [2, 0, 4, 1, 3]);  view_1627 = None
    view_1628: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_2070, [16, 512, 64]);  permute_2070 = None
    bmm_449: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_2071, view_1628);  permute_2071 = None
    bmm_450: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1628, permute_2072);  view_1628 = permute_2072 = None
    view_1629: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_449, [16, 512, 1, 64, 1]);  bmm_449 = None
    permute_2073: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1629, [4, 2, 0, 3, 1]);  view_1629 = None
    view_1630: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_450, [16, 512, 512, 1, 1]);  bmm_450 = None
    permute_2074: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1630, [1, 3, 0, 4, 2]);  view_1630 = None
    permute_2075: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2073, [4, 1, 2, 3, 0]);  permute_2073 = None
    squeeze_395: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2075, 4);  permute_2075 = None
    permute_2076: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_2074, [1, 2, 0, 4, 3]);  permute_2074 = None
    squeeze_396: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_2076, 4);  permute_2076 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_77: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_771: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_77, 1.1111111111111112);  convert_element_type_77 = None
    mul_772: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_396, mul_771);  squeeze_396 = mul_771 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_773: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_772, alias_43);  mul_772 = None
    sum_261: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_773, [3], True)
    mul_774: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_43, sum_261);  alias_43 = sum_261 = None
    sub_200: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_773, mul_774);  mul_773 = mul_774 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_775: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_200, 0.125);  sub_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_17: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_775, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1631: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_17, [1, 16, 1023, 512]);  index_put_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_69: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1631, 2, 1, 9223372036854775807);  view_1631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1632: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_69, [1, 16, 512, 1024]);  slice_scatter_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1633: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1632, [1, 16, 512, 1024, 1]);  view_1632 = None
    permute_2077: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1633, [1, 2, 4, 0, 3]);  view_1633 = None
    view_1634: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_2077, [16, 512, 1024]);  permute_2077 = None
    bmm_451: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_2078, view_1634);  permute_2078 = None
    bmm_452: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1634, permute_2079);  view_1634 = permute_2079 = None
    view_1635: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_451, [16, 64, 1, 1024, 1]);  bmm_451 = None
    permute_2080: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1635, [2, 0, 4, 3, 1]);  view_1635 = None
    view_1636: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_452, [16, 512, 64, 1, 1]);  bmm_452 = None
    permute_2081: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1636, [3, 0, 1, 4, 2]);  view_1636 = None
    permute_2082: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2080, [3, 0, 1, 4, 2]);  permute_2080 = None
    squeeze_397: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2082, 4);  permute_2082 = None
    permute_2083: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2081, [2, 0, 1, 4, 3]);  permute_2081 = None
    squeeze_398: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2083, 4);  permute_2083 = None
    sum_262: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_398, [0, 1], True)
    view_1637: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_262, [16, 64]);  sum_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1638: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_775, [1, 16, 512, 512, 1]);  mul_775 = None
    permute_2084: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1638, [1, 2, 4, 0, 3]);  view_1638 = None
    view_1639: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_2084, [16, 512, 512]);  permute_2084 = None
    bmm_453: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_2085, view_1639);  permute_2085 = None
    bmm_454: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1639, permute_2086);  view_1639 = permute_2086 = None
    view_1640: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_453, [16, 64, 1, 512, 1]);  bmm_453 = None
    permute_2087: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1640, [2, 0, 4, 3, 1]);  view_1640 = None
    view_1641: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_454, [16, 512, 64, 1, 1]);  bmm_454 = None
    permute_2088: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1641, [3, 0, 1, 4, 2]);  view_1641 = None
    permute_2089: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2087, [3, 0, 1, 4, 2]);  permute_2087 = None
    squeeze_399: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2089, 4);  permute_2089 = None
    permute_2090: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2088, [2, 0, 1, 4, 3]);  permute_2088 = None
    squeeze_400: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2090, 4);  permute_2090 = None
    sum_263: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_400, [0, 1], True)
    view_1642: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_263, [16, 64]);  sum_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_389: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_398, squeeze_400);  squeeze_398 = squeeze_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1643: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_397, [1024, 1, 16, 64, 1]);  squeeze_397 = None
    permute_2091: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1643, [0, 4, 1, 2, 3]);  view_1643 = None
    view_1644: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_2091, [1, 1024, 1024]);  permute_2091 = None
    bmm_455: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1644);  view_1644 = None
    view_1645: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_455, [1024, 1, 16, 64, 1]);  bmm_455 = None
    permute_2093: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1645, [4, 1, 2, 3, 0]);  view_1645 = None
    permute_2094: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2093, [4, 2, 3, 0, 1]);  permute_2093 = None
    squeeze_401: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2094, 4);  permute_2094 = None
    squeeze_402: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_401, 3);  squeeze_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1646: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_395, [512, 1, 16, 64, 1]);  squeeze_395 = None
    permute_2095: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1646, [0, 4, 1, 2, 3]);  view_1646 = None
    clone_155: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_2095, memory_format = torch.contiguous_format);  permute_2095 = None
    view_1647: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_155, [1, 512, 1024]);  clone_155 = None
    bmm_456: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2096, view_1647)
    bmm_457: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1647, permute_2097);  view_1647 = permute_2097 = None
    view_1648: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_456, [1024, 1, 16, 64, 1]);  bmm_456 = None
    permute_2098: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1648, [4, 1, 2, 3, 0]);  view_1648 = None
    view_1649: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_457, [512, 1024, 1, 1, 1]);  bmm_457 = None
    permute_2099: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1649, [0, 2, 3, 4, 1]);  view_1649 = None
    permute_2100: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2098, [4, 2, 3, 0, 1]);  permute_2098 = None
    squeeze_403: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2100, 4);  permute_2100 = None
    squeeze_404: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_403, 3);  squeeze_403 = None
    permute_2101: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2099, [0, 1, 4, 2, 3]);  permute_2099 = None
    squeeze_405: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2101, 4);  permute_2101 = None
    squeeze_406: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_405, 3);  squeeze_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_390: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_767, squeeze_406);  mul_767 = squeeze_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1650: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_399, [512, 1, 16, 64, 1]);  squeeze_399 = None
    permute_2102: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1650, [0, 4, 1, 2, 3]);  view_1650 = None
    view_1651: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_2102, [1, 512, 1024]);  permute_2102 = None
    bmm_458: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2096, view_1651)
    bmm_459: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1651, permute_2104);  view_1651 = permute_2104 = None
    view_1652: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_458, [1024, 1, 16, 64, 1]);  bmm_458 = None
    permute_2105: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1652, [4, 1, 2, 3, 0]);  view_1652 = None
    view_1653: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_459, [512, 1024, 1, 1, 1]);  bmm_459 = None
    permute_2106: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1653, [0, 2, 3, 4, 1]);  view_1653 = None
    permute_2107: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2105, [4, 2, 3, 0, 1]);  permute_2105 = None
    squeeze_407: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2107, 4);  permute_2107 = None
    squeeze_408: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_407, 3);  squeeze_407 = None
    permute_2108: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2106, [0, 1, 4, 2, 3]);  permute_2106 = None
    squeeze_409: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2108, 4);  permute_2108 = None
    squeeze_410: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_409, 3);  squeeze_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_391: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_390, squeeze_410);  add_390 = squeeze_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1654: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_389, [512, 1, 16, 64, 1]);  add_389 = None
    permute_2109: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1654, [0, 4, 1, 2, 3]);  view_1654 = None
    clone_156: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_2109, memory_format = torch.contiguous_format);  permute_2109 = None
    view_1655: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_156, [1, 512, 1024]);  clone_156 = None
    bmm_460: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2096, view_1655);  permute_2096 = None
    bmm_461: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1655, permute_2111);  view_1655 = permute_2111 = None
    view_1656: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_460, [1024, 1, 16, 64, 1]);  bmm_460 = None
    permute_2112: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1656, [4, 1, 2, 3, 0]);  view_1656 = None
    view_1657: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_461, [512, 1024, 1, 1, 1]);  bmm_461 = None
    permute_2113: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1657, [0, 2, 3, 4, 1]);  view_1657 = None
    permute_2114: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2112, [4, 2, 3, 0, 1]);  permute_2112 = None
    squeeze_411: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2114, 4);  permute_2114 = None
    squeeze_412: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_411, 3);  squeeze_411 = None
    permute_2115: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2113, [0, 1, 4, 2, 3]);  permute_2113 = None
    squeeze_413: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2115, 4);  permute_2115 = None
    squeeze_414: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_413, 3);  squeeze_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_392: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_391, squeeze_414);  add_391 = squeeze_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_777: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_392, primals_216);  primals_216 = None
    mul_778: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_777, 1024)
    sum_264: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_777, [2], True)
    mul_779: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_777, mul_50);  mul_777 = None
    sum_265: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_779, [2], True);  mul_779 = None
    mul_780: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_50, sum_265);  sum_265 = None
    sub_202: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_778, sum_264);  mul_778 = sum_264 = None
    sub_203: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_202, mul_780);  sub_202 = mul_780 = None
    mul_781: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_63, sub_203);  div_63 = sub_203 = None
    mul_782: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_392, mul_50);  mul_50 = None
    sum_266: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_782, [0, 1]);  mul_782 = None
    sum_267: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_392, [0, 1]);  add_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_78: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_73, torch.float32);  getitem_73 = None
    mul_783: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_78, 1.1111111111111112);  convert_element_type_78 = None
    mul_784: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_781, mul_783);  mul_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1658: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_784, [512, 1024]);  mul_784 = None
    mm_74: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1658, permute_2116);  permute_2116 = None
    permute_2117: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1658, [1, 0])
    mm_75: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_2117, view_226);  permute_2117 = view_226 = None
    permute_2118: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_268: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1658, [0], True);  view_1658 = None
    view_1659: "f32[1024]" = torch.ops.aten.reshape.default(sum_268, [1024]);  sum_268 = None
    permute_2119: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_2118, [1, 0]);  permute_2118 = None
    view_1660: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_74, [512, 1, 4096]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_79: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_785: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_79, 1.1111111111111112);  convert_element_type_79 = None
    mul_786: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1660, mul_785);  view_1660 = mul_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_788: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_64, 0.5);  add_64 = None
    mul_789: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_225, view_225)
    mul_790: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_789, -0.5);  mul_789 = None
    exp_44: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_790);  mul_790 = None
    mul_791: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
    mul_792: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_225, mul_791);  view_225 = mul_791 = None
    add_394: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_788, mul_792);  mul_788 = mul_792 = None
    mul_793: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_786, add_394);  mul_786 = add_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1661: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_793, [512, 4096]);  mul_793 = None
    mm_76: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1661, permute_2120);  permute_2120 = None
    permute_2121: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1661, [1, 0])
    mm_77: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_2121, view_224);  permute_2121 = view_224 = None
    permute_2122: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_269: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1661, [0], True);  view_1661 = None
    view_1662: "f32[4096]" = torch.ops.aten.reshape.default(sum_269, [4096]);  sum_269 = None
    permute_2123: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_2122, [1, 0]);  permute_2122 = None
    view_1663: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_76, [512, 1, 1024]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_395: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_781, view_1663);  mul_781 = view_1663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_795: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_395, primals_210);  primals_210 = None
    mul_796: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_795, 1024)
    sum_270: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_795, [2], True)
    mul_797: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_795, mul_45);  mul_795 = None
    sum_271: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_797, [2], True);  mul_797 = None
    mul_798: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_45, sum_271);  sum_271 = None
    sub_205: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_796, sum_270);  mul_796 = sum_270 = None
    sub_206: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_205, mul_798);  sub_205 = mul_798 = None
    mul_799: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_64, sub_206);  div_64 = sub_206 = None
    mul_800: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_395, mul_45);  mul_45 = None
    sum_272: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_800, [0, 1]);  mul_800 = None
    sum_273: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_395, [0, 1]);  add_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_80: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_801: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_80, 1.1111111111111112);  convert_element_type_80 = None
    mul_802: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_799, mul_801);  mul_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1664: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_802, [512, 1, 1024, 1, 1]);  mul_802 = None
    permute_2124: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1664, [0, 3, 4, 1, 2]);  view_1664 = None
    view_1665: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_2124, [1, 512, 1024]);  permute_2124 = None
    bmm_462: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2125, view_1665);  permute_2125 = None
    bmm_463: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1665, permute_2126);  view_1665 = permute_2126 = None
    view_1666: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_462, [64, 16, 1, 1024, 1]);  bmm_462 = None
    permute_2127: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1666, [4, 2, 3, 0, 1]);  view_1666 = None
    view_1667: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_463, [512, 64, 16, 1, 1]);  bmm_463 = None
    permute_2128: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1667, [0, 3, 4, 1, 2]);  view_1667 = None
    permute_2129: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2127, [2, 4, 3, 0, 1]);  permute_2127 = None
    squeeze_415: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2129, 4);  permute_2129 = None
    squeeze_416: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_415, 3);  squeeze_415 = None
    permute_2130: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2128, [0, 1, 4, 3, 2]);  permute_2128 = None
    squeeze_417: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2130, 4);  permute_2130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1668: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_417, [512, 1, 16, 64, 1]);  squeeze_417 = None
    permute_2131: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1668, [2, 0, 4, 1, 3]);  view_1668 = None
    view_1669: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_2131, [16, 512, 64]);  permute_2131 = None
    bmm_464: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_2132, view_1669);  permute_2132 = None
    bmm_465: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1669, permute_2133);  view_1669 = permute_2133 = None
    view_1670: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_464, [16, 512, 1, 64, 1]);  bmm_464 = None
    permute_2134: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1670, [4, 2, 0, 3, 1]);  view_1670 = None
    view_1671: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_465, [16, 512, 512, 1, 1]);  bmm_465 = None
    permute_2135: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1671, [1, 3, 0, 4, 2]);  view_1671 = None
    permute_2136: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2134, [4, 1, 2, 3, 0]);  permute_2134 = None
    squeeze_418: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2136, 4);  permute_2136 = None
    permute_2137: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_2135, [1, 2, 0, 4, 3]);  permute_2135 = None
    squeeze_419: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_2137, 4);  permute_2137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_81: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_65, torch.float32);  getitem_65 = None
    mul_803: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_81, 1.1111111111111112);  convert_element_type_81 = None
    mul_804: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_419, mul_803);  squeeze_419 = mul_803 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_805: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_804, alias_44);  mul_804 = None
    sum_274: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_805, [3], True)
    mul_806: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_44, sum_274);  alias_44 = sum_274 = None
    sub_207: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_805, mul_806);  mul_805 = mul_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_807: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_207, 0.125);  sub_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_18: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_807, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1672: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_18, [1, 16, 1023, 512]);  index_put_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_73: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1672, 2, 1, 9223372036854775807);  view_1672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1673: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_73, [1, 16, 512, 1024]);  slice_scatter_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1674: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1673, [1, 16, 512, 1024, 1]);  view_1673 = None
    permute_2138: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1674, [1, 2, 4, 0, 3]);  view_1674 = None
    view_1675: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_2138, [16, 512, 1024]);  permute_2138 = None
    bmm_466: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_2139, view_1675);  permute_2139 = None
    bmm_467: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1675, permute_2140);  view_1675 = permute_2140 = None
    view_1676: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_466, [16, 64, 1, 1024, 1]);  bmm_466 = None
    permute_2141: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1676, [2, 0, 4, 3, 1]);  view_1676 = None
    view_1677: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_467, [16, 512, 64, 1, 1]);  bmm_467 = None
    permute_2142: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1677, [3, 0, 1, 4, 2]);  view_1677 = None
    permute_2143: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2141, [3, 0, 1, 4, 2]);  permute_2141 = None
    squeeze_420: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2143, 4);  permute_2143 = None
    permute_2144: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2142, [2, 0, 1, 4, 3]);  permute_2142 = None
    squeeze_421: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2144, 4);  permute_2144 = None
    sum_275: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_421, [0, 1], True)
    view_1678: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_275, [16, 64]);  sum_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1679: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_807, [1, 16, 512, 512, 1]);  mul_807 = None
    permute_2145: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1679, [1, 2, 4, 0, 3]);  view_1679 = None
    view_1680: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_2145, [16, 512, 512]);  permute_2145 = None
    bmm_468: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_2146, view_1680);  permute_2146 = None
    bmm_469: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1680, permute_2147);  view_1680 = permute_2147 = None
    view_1681: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_468, [16, 64, 1, 512, 1]);  bmm_468 = None
    permute_2148: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1681, [2, 0, 4, 3, 1]);  view_1681 = None
    view_1682: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_469, [16, 512, 64, 1, 1]);  bmm_469 = None
    permute_2149: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1682, [3, 0, 1, 4, 2]);  view_1682 = None
    permute_2150: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2148, [3, 0, 1, 4, 2]);  permute_2148 = None
    squeeze_422: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2150, 4);  permute_2150 = None
    permute_2151: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2149, [2, 0, 1, 4, 3]);  permute_2149 = None
    squeeze_423: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2151, 4);  permute_2151 = None
    sum_276: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_423, [0, 1], True)
    view_1683: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_276, [16, 64]);  sum_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_396: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_421, squeeze_423);  squeeze_421 = squeeze_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1684: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_420, [1024, 1, 16, 64, 1]);  squeeze_420 = None
    permute_2152: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1684, [0, 4, 1, 2, 3]);  view_1684 = None
    view_1685: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_2152, [1, 1024, 1024]);  permute_2152 = None
    bmm_470: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1685);  view_1685 = None
    view_1686: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_470, [1024, 1, 16, 64, 1]);  bmm_470 = None
    permute_2154: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1686, [4, 1, 2, 3, 0]);  view_1686 = None
    permute_2155: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2154, [4, 2, 3, 0, 1]);  permute_2154 = None
    squeeze_424: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2155, 4);  permute_2155 = None
    squeeze_425: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_424, 3);  squeeze_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1687: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_418, [512, 1, 16, 64, 1]);  squeeze_418 = None
    permute_2156: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1687, [0, 4, 1, 2, 3]);  view_1687 = None
    clone_161: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_2156, memory_format = torch.contiguous_format);  permute_2156 = None
    view_1688: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_161, [1, 512, 1024]);  clone_161 = None
    bmm_471: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2157, view_1688)
    bmm_472: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1688, permute_2158);  view_1688 = permute_2158 = None
    view_1689: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_471, [1024, 1, 16, 64, 1]);  bmm_471 = None
    permute_2159: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1689, [4, 1, 2, 3, 0]);  view_1689 = None
    view_1690: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_472, [512, 1024, 1, 1, 1]);  bmm_472 = None
    permute_2160: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1690, [0, 2, 3, 4, 1]);  view_1690 = None
    permute_2161: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2159, [4, 2, 3, 0, 1]);  permute_2159 = None
    squeeze_426: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2161, 4);  permute_2161 = None
    squeeze_427: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_426, 3);  squeeze_426 = None
    permute_2162: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2160, [0, 1, 4, 2, 3]);  permute_2160 = None
    squeeze_428: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2162, 4);  permute_2162 = None
    squeeze_429: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_428, 3);  squeeze_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_397: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_799, squeeze_429);  mul_799 = squeeze_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1691: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_422, [512, 1, 16, 64, 1]);  squeeze_422 = None
    permute_2163: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1691, [0, 4, 1, 2, 3]);  view_1691 = None
    view_1692: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_2163, [1, 512, 1024]);  permute_2163 = None
    bmm_473: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2157, view_1692)
    bmm_474: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1692, permute_2165);  view_1692 = permute_2165 = None
    view_1693: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_473, [1024, 1, 16, 64, 1]);  bmm_473 = None
    permute_2166: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1693, [4, 1, 2, 3, 0]);  view_1693 = None
    view_1694: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_474, [512, 1024, 1, 1, 1]);  bmm_474 = None
    permute_2167: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1694, [0, 2, 3, 4, 1]);  view_1694 = None
    permute_2168: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2166, [4, 2, 3, 0, 1]);  permute_2166 = None
    squeeze_430: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2168, 4);  permute_2168 = None
    squeeze_431: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_430, 3);  squeeze_430 = None
    permute_2169: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2167, [0, 1, 4, 2, 3]);  permute_2167 = None
    squeeze_432: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2169, 4);  permute_2169 = None
    squeeze_433: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_432, 3);  squeeze_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_398: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_397, squeeze_433);  add_397 = squeeze_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1695: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_396, [512, 1, 16, 64, 1]);  add_396 = None
    permute_2170: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1695, [0, 4, 1, 2, 3]);  view_1695 = None
    clone_162: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_2170, memory_format = torch.contiguous_format);  permute_2170 = None
    view_1696: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_162, [1, 512, 1024]);  clone_162 = None
    bmm_475: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2157, view_1696);  permute_2157 = None
    bmm_476: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1696, permute_2172);  view_1696 = permute_2172 = None
    view_1697: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_475, [1024, 1, 16, 64, 1]);  bmm_475 = None
    permute_2173: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1697, [4, 1, 2, 3, 0]);  view_1697 = None
    view_1698: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_476, [512, 1024, 1, 1, 1]);  bmm_476 = None
    permute_2174: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1698, [0, 2, 3, 4, 1]);  view_1698 = None
    permute_2175: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2173, [4, 2, 3, 0, 1]);  permute_2173 = None
    squeeze_434: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2175, 4);  permute_2175 = None
    squeeze_435: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_434, 3);  squeeze_434 = None
    permute_2176: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2174, [0, 1, 4, 2, 3]);  permute_2174 = None
    squeeze_436: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2176, 4);  permute_2176 = None
    squeeze_437: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_436, 3);  squeeze_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_399: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_398, squeeze_437);  add_398 = squeeze_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_809: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_399, primals_208);  primals_208 = None
    mul_810: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_809, 1024)
    sum_277: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_809, [2], True)
    mul_811: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_809, mul_42);  mul_809 = None
    sum_278: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_811, [2], True);  mul_811 = None
    mul_812: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_42, sum_278);  sum_278 = None
    sub_209: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_810, sum_277);  mul_810 = sum_277 = None
    sub_210: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_209, mul_812);  sub_209 = mul_812 = None
    mul_813: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_65, sub_210);  div_65 = sub_210 = None
    mul_814: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_399, mul_42);  mul_42 = None
    sum_279: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 1]);  mul_814 = None
    sum_280: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_399, [0, 1]);  add_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_82: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_815: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_82, 1.1111111111111112);  convert_element_type_82 = None
    mul_816: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_813, mul_815);  mul_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1699: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_816, [512, 1024]);  mul_816 = None
    mm_78: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1699, permute_2177);  permute_2177 = None
    permute_2178: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1699, [1, 0])
    mm_79: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_2178, view_188);  permute_2178 = view_188 = None
    permute_2179: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_281: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1699, [0], True);  view_1699 = None
    view_1700: "f32[1024]" = torch.ops.aten.reshape.default(sum_281, [1024]);  sum_281 = None
    permute_2180: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_2179, [1, 0]);  permute_2179 = None
    view_1701: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_78, [512, 1, 4096]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_83: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_59, torch.float32);  getitem_59 = None
    mul_817: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_83, 1.1111111111111112);  convert_element_type_83 = None
    mul_818: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1701, mul_817);  view_1701 = mul_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_820: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_53, 0.5);  add_53 = None
    mul_821: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_187, view_187)
    mul_822: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_821, -0.5);  mul_821 = None
    exp_45: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_822);  mul_822 = None
    mul_823: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
    mul_824: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_187, mul_823);  view_187 = mul_823 = None
    add_401: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_820, mul_824);  mul_820 = mul_824 = None
    mul_825: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_818, add_401);  mul_818 = add_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1702: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_825, [512, 4096]);  mul_825 = None
    mm_80: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1702, permute_2181);  permute_2181 = None
    permute_2182: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1702, [1, 0])
    mm_81: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_2182, view_186);  permute_2182 = view_186 = None
    permute_2183: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_282: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1702, [0], True);  view_1702 = None
    view_1703: "f32[4096]" = torch.ops.aten.reshape.default(sum_282, [4096]);  sum_282 = None
    permute_2184: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_2183, [1, 0]);  permute_2183 = None
    view_1704: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_80, [512, 1, 1024]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_402: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_813, view_1704);  mul_813 = view_1704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_827: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_402, primals_202);  primals_202 = None
    mul_828: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_827, 1024)
    sum_283: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_827, [2], True)
    mul_829: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_827, mul_37);  mul_827 = None
    sum_284: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_829, [2], True);  mul_829 = None
    mul_830: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_37, sum_284);  sum_284 = None
    sub_212: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_828, sum_283);  mul_828 = sum_283 = None
    sub_213: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_212, mul_830);  sub_212 = mul_830 = None
    mul_831: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_66, sub_213);  div_66 = sub_213 = None
    mul_832: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_402, mul_37);  mul_37 = None
    sum_285: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_832, [0, 1]);  mul_832 = None
    sum_286: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_402, [0, 1]);  add_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_84: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_55, torch.float32);  getitem_55 = None
    mul_833: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_84, 1.1111111111111112);  convert_element_type_84 = None
    mul_834: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_831, mul_833);  mul_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1705: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_834, [512, 1, 1024, 1, 1]);  mul_834 = None
    permute_2185: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1705, [0, 3, 4, 1, 2]);  view_1705 = None
    view_1706: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_2185, [1, 512, 1024]);  permute_2185 = None
    bmm_477: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2186, view_1706);  permute_2186 = None
    bmm_478: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1706, permute_2187);  view_1706 = permute_2187 = None
    view_1707: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_477, [64, 16, 1, 1024, 1]);  bmm_477 = None
    permute_2188: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1707, [4, 2, 3, 0, 1]);  view_1707 = None
    view_1708: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_478, [512, 64, 16, 1, 1]);  bmm_478 = None
    permute_2189: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1708, [0, 3, 4, 1, 2]);  view_1708 = None
    permute_2190: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2188, [2, 4, 3, 0, 1]);  permute_2188 = None
    squeeze_438: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2190, 4);  permute_2190 = None
    squeeze_439: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_438, 3);  squeeze_438 = None
    permute_2191: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2189, [0, 1, 4, 3, 2]);  permute_2189 = None
    squeeze_440: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2191, 4);  permute_2191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1709: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_440, [512, 1, 16, 64, 1]);  squeeze_440 = None
    permute_2192: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1709, [2, 0, 4, 1, 3]);  view_1709 = None
    view_1710: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_2192, [16, 512, 64]);  permute_2192 = None
    bmm_479: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_2193, view_1710);  permute_2193 = None
    bmm_480: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1710, permute_2194);  view_1710 = permute_2194 = None
    view_1711: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_479, [16, 512, 1, 64, 1]);  bmm_479 = None
    permute_2195: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1711, [4, 2, 0, 3, 1]);  view_1711 = None
    view_1712: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_480, [16, 512, 512, 1, 1]);  bmm_480 = None
    permute_2196: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1712, [1, 3, 0, 4, 2]);  view_1712 = None
    permute_2197: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2195, [4, 1, 2, 3, 0]);  permute_2195 = None
    squeeze_441: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2197, 4);  permute_2197 = None
    permute_2198: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_2196, [1, 2, 0, 4, 3]);  permute_2196 = None
    squeeze_442: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_2198, 4);  permute_2198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_85: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_53, torch.float32);  getitem_53 = None
    mul_835: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_85, 1.1111111111111112);  convert_element_type_85 = None
    mul_836: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_442, mul_835);  squeeze_442 = mul_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_837: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_836, alias_45);  mul_836 = None
    sum_287: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_837, [3], True)
    mul_838: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_45, sum_287);  alias_45 = sum_287 = None
    sub_214: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_837, mul_838);  mul_837 = mul_838 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_839: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_214, 0.125);  sub_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_19: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_839, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1713: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_19, [1, 16, 1023, 512]);  index_put_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_77: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1713, 2, 1, 9223372036854775807);  view_1713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1714: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_77, [1, 16, 512, 1024]);  slice_scatter_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1715: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1714, [1, 16, 512, 1024, 1]);  view_1714 = None
    permute_2199: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1715, [1, 2, 4, 0, 3]);  view_1715 = None
    view_1716: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_2199, [16, 512, 1024]);  permute_2199 = None
    bmm_481: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_2200, view_1716);  permute_2200 = None
    bmm_482: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1716, permute_2201);  view_1716 = permute_2201 = None
    view_1717: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_481, [16, 64, 1, 1024, 1]);  bmm_481 = None
    permute_2202: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1717, [2, 0, 4, 3, 1]);  view_1717 = None
    view_1718: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_482, [16, 512, 64, 1, 1]);  bmm_482 = None
    permute_2203: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1718, [3, 0, 1, 4, 2]);  view_1718 = None
    permute_2204: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2202, [3, 0, 1, 4, 2]);  permute_2202 = None
    squeeze_443: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2204, 4);  permute_2204 = None
    permute_2205: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2203, [2, 0, 1, 4, 3]);  permute_2203 = None
    squeeze_444: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2205, 4);  permute_2205 = None
    sum_288: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_444, [0, 1], True)
    view_1719: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_288, [16, 64]);  sum_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1720: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_839, [1, 16, 512, 512, 1]);  mul_839 = None
    permute_2206: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1720, [1, 2, 4, 0, 3]);  view_1720 = None
    view_1721: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_2206, [16, 512, 512]);  permute_2206 = None
    bmm_483: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_2207, view_1721);  permute_2207 = None
    bmm_484: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1721, permute_2208);  view_1721 = permute_2208 = None
    view_1722: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_483, [16, 64, 1, 512, 1]);  bmm_483 = None
    permute_2209: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1722, [2, 0, 4, 3, 1]);  view_1722 = None
    view_1723: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_484, [16, 512, 64, 1, 1]);  bmm_484 = None
    permute_2210: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1723, [3, 0, 1, 4, 2]);  view_1723 = None
    permute_2211: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2209, [3, 0, 1, 4, 2]);  permute_2209 = None
    squeeze_445: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2211, 4);  permute_2211 = None
    permute_2212: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2210, [2, 0, 1, 4, 3]);  permute_2210 = None
    squeeze_446: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2212, 4);  permute_2212 = None
    sum_289: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_446, [0, 1], True)
    view_1724: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_289, [16, 64]);  sum_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_403: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_444, squeeze_446);  squeeze_444 = squeeze_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1725: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_443, [1024, 1, 16, 64, 1]);  squeeze_443 = None
    permute_2213: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1725, [0, 4, 1, 2, 3]);  view_1725 = None
    view_1726: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_2213, [1, 1024, 1024]);  permute_2213 = None
    bmm_485: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1726);  view_1726 = None
    view_1727: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_485, [1024, 1, 16, 64, 1]);  bmm_485 = None
    permute_2215: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1727, [4, 1, 2, 3, 0]);  view_1727 = None
    permute_2216: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2215, [4, 2, 3, 0, 1]);  permute_2215 = None
    squeeze_447: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2216, 4);  permute_2216 = None
    squeeze_448: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_447, 3);  squeeze_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1728: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_441, [512, 1, 16, 64, 1]);  squeeze_441 = None
    permute_2217: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1728, [0, 4, 1, 2, 3]);  view_1728 = None
    clone_167: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_2217, memory_format = torch.contiguous_format);  permute_2217 = None
    view_1729: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_167, [1, 512, 1024]);  clone_167 = None
    bmm_486: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2218, view_1729)
    bmm_487: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1729, permute_2219);  view_1729 = permute_2219 = None
    view_1730: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_486, [1024, 1, 16, 64, 1]);  bmm_486 = None
    permute_2220: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1730, [4, 1, 2, 3, 0]);  view_1730 = None
    view_1731: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_487, [512, 1024, 1, 1, 1]);  bmm_487 = None
    permute_2221: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1731, [0, 2, 3, 4, 1]);  view_1731 = None
    permute_2222: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2220, [4, 2, 3, 0, 1]);  permute_2220 = None
    squeeze_449: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2222, 4);  permute_2222 = None
    squeeze_450: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_449, 3);  squeeze_449 = None
    permute_2223: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2221, [0, 1, 4, 2, 3]);  permute_2221 = None
    squeeze_451: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2223, 4);  permute_2223 = None
    squeeze_452: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_451, 3);  squeeze_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_404: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_831, squeeze_452);  mul_831 = squeeze_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1732: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_445, [512, 1, 16, 64, 1]);  squeeze_445 = None
    permute_2224: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1732, [0, 4, 1, 2, 3]);  view_1732 = None
    view_1733: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_2224, [1, 512, 1024]);  permute_2224 = None
    bmm_488: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2218, view_1733)
    bmm_489: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1733, permute_2226);  view_1733 = permute_2226 = None
    view_1734: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_488, [1024, 1, 16, 64, 1]);  bmm_488 = None
    permute_2227: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1734, [4, 1, 2, 3, 0]);  view_1734 = None
    view_1735: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_489, [512, 1024, 1, 1, 1]);  bmm_489 = None
    permute_2228: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1735, [0, 2, 3, 4, 1]);  view_1735 = None
    permute_2229: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2227, [4, 2, 3, 0, 1]);  permute_2227 = None
    squeeze_453: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2229, 4);  permute_2229 = None
    squeeze_454: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_453, 3);  squeeze_453 = None
    permute_2230: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2228, [0, 1, 4, 2, 3]);  permute_2228 = None
    squeeze_455: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2230, 4);  permute_2230 = None
    squeeze_456: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_455, 3);  squeeze_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_405: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_404, squeeze_456);  add_404 = squeeze_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1736: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_403, [512, 1, 16, 64, 1]);  add_403 = None
    permute_2231: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1736, [0, 4, 1, 2, 3]);  view_1736 = None
    clone_168: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_2231, memory_format = torch.contiguous_format);  permute_2231 = None
    view_1737: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_168, [1, 512, 1024]);  clone_168 = None
    bmm_490: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2218, view_1737);  permute_2218 = None
    bmm_491: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1737, permute_2233);  view_1737 = permute_2233 = None
    view_1738: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_490, [1024, 1, 16, 64, 1]);  bmm_490 = None
    permute_2234: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1738, [4, 1, 2, 3, 0]);  view_1738 = None
    view_1739: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_491, [512, 1024, 1, 1, 1]);  bmm_491 = None
    permute_2235: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1739, [0, 2, 3, 4, 1]);  view_1739 = None
    permute_2236: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2234, [4, 2, 3, 0, 1]);  permute_2234 = None
    squeeze_457: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2236, 4);  permute_2236 = None
    squeeze_458: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_457, 3);  squeeze_457 = None
    permute_2237: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2235, [0, 1, 4, 2, 3]);  permute_2235 = None
    squeeze_459: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2237, 4);  permute_2237 = None
    squeeze_460: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_459, 3);  squeeze_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_406: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_405, squeeze_460);  add_405 = squeeze_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_841: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_406, primals_200);  primals_200 = None
    mul_842: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_841, 1024)
    sum_290: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_841, [2], True)
    mul_843: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_841, mul_34);  mul_841 = None
    sum_291: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_843, [2], True);  mul_843 = None
    mul_844: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_34, sum_291);  sum_291 = None
    sub_216: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_842, sum_290);  mul_842 = sum_290 = None
    sub_217: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_216, mul_844);  sub_216 = mul_844 = None
    mul_845: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_67, sub_217);  div_67 = sub_217 = None
    mul_846: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_406, mul_34);  mul_34 = None
    sum_292: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_846, [0, 1]);  mul_846 = None
    sum_293: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_406, [0, 1]);  add_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_86: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_49, torch.float32);  getitem_49 = None
    mul_847: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_86, 1.1111111111111112);  convert_element_type_86 = None
    mul_848: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_845, mul_847);  mul_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1740: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_848, [512, 1024]);  mul_848 = None
    mm_82: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1740, permute_2238);  permute_2238 = None
    permute_2239: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1740, [1, 0])
    mm_83: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_2239, view_150);  permute_2239 = view_150 = None
    permute_2240: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_294: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1740, [0], True);  view_1740 = None
    view_1741: "f32[1024]" = torch.ops.aten.reshape.default(sum_294, [1024]);  sum_294 = None
    permute_2241: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_2240, [1, 0]);  permute_2240 = None
    view_1742: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_82, [512, 1, 4096]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_87: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_849: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_87, 1.1111111111111112);  convert_element_type_87 = None
    mul_850: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1742, mul_849);  view_1742 = mul_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_852: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_42, 0.5);  add_42 = None
    mul_853: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_149, view_149)
    mul_854: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_853, -0.5);  mul_853 = None
    exp_46: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_854);  mul_854 = None
    mul_855: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
    mul_856: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_149, mul_855);  view_149 = mul_855 = None
    add_408: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_852, mul_856);  mul_852 = mul_856 = None
    mul_857: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_850, add_408);  mul_850 = add_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1743: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_857, [512, 4096]);  mul_857 = None
    mm_84: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1743, permute_2242);  permute_2242 = None
    permute_2243: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1743, [1, 0])
    mm_85: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_2243, view_148);  permute_2243 = view_148 = None
    permute_2244: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_295: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1743, [0], True);  view_1743 = None
    view_1744: "f32[4096]" = torch.ops.aten.reshape.default(sum_295, [4096]);  sum_295 = None
    permute_2245: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_2244, [1, 0]);  permute_2244 = None
    view_1745: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_84, [512, 1, 1024]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_409: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_845, view_1745);  mul_845 = view_1745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_859: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_409, primals_194);  primals_194 = None
    mul_860: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_859, 1024)
    sum_296: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_859, [2], True)
    mul_861: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_859, mul_29);  mul_859 = None
    sum_297: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_861, [2], True);  mul_861 = None
    mul_862: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_29, sum_297);  sum_297 = None
    sub_219: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_860, sum_296);  mul_860 = sum_296 = None
    sub_220: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_219, mul_862);  sub_219 = mul_862 = None
    mul_863: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_68, sub_220);  div_68 = sub_220 = None
    mul_864: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_409, mul_29);  mul_29 = None
    sum_298: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_864, [0, 1]);  mul_864 = None
    sum_299: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_409, [0, 1]);  add_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_88: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_43, torch.float32);  getitem_43 = None
    mul_865: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_88, 1.1111111111111112);  convert_element_type_88 = None
    mul_866: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_863, mul_865);  mul_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1746: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_866, [512, 1, 1024, 1, 1]);  mul_866 = None
    permute_2246: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1746, [0, 3, 4, 1, 2]);  view_1746 = None
    view_1747: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_2246, [1, 512, 1024]);  permute_2246 = None
    bmm_492: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2247, view_1747);  permute_2247 = None
    bmm_493: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1747, permute_2248);  view_1747 = permute_2248 = None
    view_1748: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_492, [64, 16, 1, 1024, 1]);  bmm_492 = None
    permute_2249: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1748, [4, 2, 3, 0, 1]);  view_1748 = None
    view_1749: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_493, [512, 64, 16, 1, 1]);  bmm_493 = None
    permute_2250: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1749, [0, 3, 4, 1, 2]);  view_1749 = None
    permute_2251: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2249, [2, 4, 3, 0, 1]);  permute_2249 = None
    squeeze_461: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2251, 4);  permute_2251 = None
    squeeze_462: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_461, 3);  squeeze_461 = None
    permute_2252: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2250, [0, 1, 4, 3, 2]);  permute_2250 = None
    squeeze_463: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2252, 4);  permute_2252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1750: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_463, [512, 1, 16, 64, 1]);  squeeze_463 = None
    permute_2253: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1750, [2, 0, 4, 1, 3]);  view_1750 = None
    view_1751: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_2253, [16, 512, 64]);  permute_2253 = None
    bmm_494: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_2254, view_1751);  permute_2254 = None
    bmm_495: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1751, permute_2255);  view_1751 = permute_2255 = None
    view_1752: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_494, [16, 512, 1, 64, 1]);  bmm_494 = None
    permute_2256: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1752, [4, 2, 0, 3, 1]);  view_1752 = None
    view_1753: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_495, [16, 512, 512, 1, 1]);  bmm_495 = None
    permute_2257: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1753, [1, 3, 0, 4, 2]);  view_1753 = None
    permute_2258: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2256, [4, 1, 2, 3, 0]);  permute_2256 = None
    squeeze_464: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2258, 4);  permute_2258 = None
    permute_2259: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_2257, [1, 2, 0, 4, 3]);  permute_2257 = None
    squeeze_465: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_2259, 4);  permute_2259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_89: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_867: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_89, 1.1111111111111112);  convert_element_type_89 = None
    mul_868: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_465, mul_867);  squeeze_465 = mul_867 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_869: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_868, alias_46);  mul_868 = None
    sum_300: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_869, [3], True)
    mul_870: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_46, sum_300);  alias_46 = sum_300 = None
    sub_221: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_869, mul_870);  mul_869 = mul_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_871: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_221, 0.125);  sub_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_20: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_871, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1754: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_20, [1, 16, 1023, 512]);  index_put_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_81: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1754, 2, 1, 9223372036854775807);  view_1754 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1755: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_81, [1, 16, 512, 1024]);  slice_scatter_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1756: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1755, [1, 16, 512, 1024, 1]);  view_1755 = None
    permute_2260: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1756, [1, 2, 4, 0, 3]);  view_1756 = None
    view_1757: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_2260, [16, 512, 1024]);  permute_2260 = None
    bmm_496: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_2261, view_1757);  permute_2261 = None
    bmm_497: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1757, permute_2262);  view_1757 = permute_2262 = None
    view_1758: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_496, [16, 64, 1, 1024, 1]);  bmm_496 = None
    permute_2263: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1758, [2, 0, 4, 3, 1]);  view_1758 = None
    view_1759: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_497, [16, 512, 64, 1, 1]);  bmm_497 = None
    permute_2264: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1759, [3, 0, 1, 4, 2]);  view_1759 = None
    permute_2265: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2263, [3, 0, 1, 4, 2]);  permute_2263 = None
    squeeze_466: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2265, 4);  permute_2265 = None
    permute_2266: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2264, [2, 0, 1, 4, 3]);  permute_2264 = None
    squeeze_467: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2266, 4);  permute_2266 = None
    sum_301: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_467, [0, 1], True)
    view_1760: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_301, [16, 64]);  sum_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1761: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_871, [1, 16, 512, 512, 1]);  mul_871 = None
    permute_2267: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1761, [1, 2, 4, 0, 3]);  view_1761 = None
    view_1762: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_2267, [16, 512, 512]);  permute_2267 = None
    bmm_498: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_2268, view_1762);  permute_2268 = None
    bmm_499: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1762, permute_2269);  view_1762 = permute_2269 = None
    view_1763: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_498, [16, 64, 1, 512, 1]);  bmm_498 = None
    permute_2270: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1763, [2, 0, 4, 3, 1]);  view_1763 = None
    view_1764: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_499, [16, 512, 64, 1, 1]);  bmm_499 = None
    permute_2271: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1764, [3, 0, 1, 4, 2]);  view_1764 = None
    permute_2272: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2270, [3, 0, 1, 4, 2]);  permute_2270 = None
    squeeze_468: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2272, 4);  permute_2272 = None
    permute_2273: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2271, [2, 0, 1, 4, 3]);  permute_2271 = None
    squeeze_469: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2273, 4);  permute_2273 = None
    sum_302: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_469, [0, 1], True)
    view_1765: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_302, [16, 64]);  sum_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_410: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_467, squeeze_469);  squeeze_467 = squeeze_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1766: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_466, [1024, 1, 16, 64, 1]);  squeeze_466 = None
    permute_2274: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1766, [0, 4, 1, 2, 3]);  view_1766 = None
    view_1767: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_2274, [1, 1024, 1024]);  permute_2274 = None
    bmm_500: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1767);  view_1767 = None
    view_1768: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_500, [1024, 1, 16, 64, 1]);  bmm_500 = None
    permute_2276: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1768, [4, 1, 2, 3, 0]);  view_1768 = None
    permute_2277: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2276, [4, 2, 3, 0, 1]);  permute_2276 = None
    squeeze_470: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2277, 4);  permute_2277 = None
    squeeze_471: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_470, 3);  squeeze_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1769: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_464, [512, 1, 16, 64, 1]);  squeeze_464 = None
    permute_2278: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1769, [0, 4, 1, 2, 3]);  view_1769 = None
    clone_173: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_2278, memory_format = torch.contiguous_format);  permute_2278 = None
    view_1770: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_173, [1, 512, 1024]);  clone_173 = None
    bmm_501: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2279, view_1770)
    bmm_502: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1770, permute_2280);  view_1770 = permute_2280 = None
    view_1771: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_501, [1024, 1, 16, 64, 1]);  bmm_501 = None
    permute_2281: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1771, [4, 1, 2, 3, 0]);  view_1771 = None
    view_1772: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_502, [512, 1024, 1, 1, 1]);  bmm_502 = None
    permute_2282: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1772, [0, 2, 3, 4, 1]);  view_1772 = None
    permute_2283: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2281, [4, 2, 3, 0, 1]);  permute_2281 = None
    squeeze_472: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2283, 4);  permute_2283 = None
    squeeze_473: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_472, 3);  squeeze_472 = None
    permute_2284: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2282, [0, 1, 4, 2, 3]);  permute_2282 = None
    squeeze_474: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2284, 4);  permute_2284 = None
    squeeze_475: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_474, 3);  squeeze_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_411: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_863, squeeze_475);  mul_863 = squeeze_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1773: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_468, [512, 1, 16, 64, 1]);  squeeze_468 = None
    permute_2285: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1773, [0, 4, 1, 2, 3]);  view_1773 = None
    view_1774: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_2285, [1, 512, 1024]);  permute_2285 = None
    bmm_503: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2279, view_1774)
    bmm_504: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1774, permute_2287);  view_1774 = permute_2287 = None
    view_1775: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_503, [1024, 1, 16, 64, 1]);  bmm_503 = None
    permute_2288: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1775, [4, 1, 2, 3, 0]);  view_1775 = None
    view_1776: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_504, [512, 1024, 1, 1, 1]);  bmm_504 = None
    permute_2289: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1776, [0, 2, 3, 4, 1]);  view_1776 = None
    permute_2290: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2288, [4, 2, 3, 0, 1]);  permute_2288 = None
    squeeze_476: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2290, 4);  permute_2290 = None
    squeeze_477: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_476, 3);  squeeze_476 = None
    permute_2291: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2289, [0, 1, 4, 2, 3]);  permute_2289 = None
    squeeze_478: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2291, 4);  permute_2291 = None
    squeeze_479: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_478, 3);  squeeze_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_412: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_411, squeeze_479);  add_411 = squeeze_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1777: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_410, [512, 1, 16, 64, 1]);  add_410 = None
    permute_2292: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1777, [0, 4, 1, 2, 3]);  view_1777 = None
    clone_174: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_2292, memory_format = torch.contiguous_format);  permute_2292 = None
    view_1778: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_174, [1, 512, 1024]);  clone_174 = None
    bmm_505: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2279, view_1778);  permute_2279 = None
    bmm_506: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1778, permute_2294);  view_1778 = permute_2294 = None
    view_1779: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_505, [1024, 1, 16, 64, 1]);  bmm_505 = None
    permute_2295: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1779, [4, 1, 2, 3, 0]);  view_1779 = None
    view_1780: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_506, [512, 1024, 1, 1, 1]);  bmm_506 = None
    permute_2296: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1780, [0, 2, 3, 4, 1]);  view_1780 = None
    permute_2297: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2295, [4, 2, 3, 0, 1]);  permute_2295 = None
    squeeze_480: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2297, 4);  permute_2297 = None
    squeeze_481: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_480, 3);  squeeze_480 = None
    permute_2298: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2296, [0, 1, 4, 2, 3]);  permute_2296 = None
    squeeze_482: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2298, 4);  permute_2298 = None
    squeeze_483: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_482, 3);  squeeze_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_413: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_412, squeeze_483);  add_412 = squeeze_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_873: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_413, primals_192);  primals_192 = None
    mul_874: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_873, 1024)
    sum_303: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_873, [2], True)
    mul_875: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_873, mul_26);  mul_873 = None
    sum_304: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_875, [2], True);  mul_875 = None
    mul_876: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_26, sum_304);  sum_304 = None
    sub_223: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_874, sum_303);  mul_874 = sum_303 = None
    sub_224: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_223, mul_876);  sub_223 = mul_876 = None
    mul_877: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_69, sub_224);  div_69 = sub_224 = None
    mul_878: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_413, mul_26);  mul_26 = None
    sum_305: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_878, [0, 1]);  mul_878 = None
    sum_306: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_413, [0, 1]);  add_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_90: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_879: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_90, 1.1111111111111112);  convert_element_type_90 = None
    mul_880: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_877, mul_879);  mul_879 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1781: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_880, [512, 1024]);  mul_880 = None
    mm_86: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1781, permute_2299);  permute_2299 = None
    permute_2300: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1781, [1, 0])
    mm_87: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_2300, view_112);  permute_2300 = view_112 = None
    permute_2301: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_307: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1781, [0], True);  view_1781 = None
    view_1782: "f32[1024]" = torch.ops.aten.reshape.default(sum_307, [1024]);  sum_307 = None
    permute_2302: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_2301, [1, 0]);  permute_2301 = None
    view_1783: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_86, [512, 1, 4096]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_91: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_35, torch.float32);  getitem_35 = None
    mul_881: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_91, 1.1111111111111112);  convert_element_type_91 = None
    mul_882: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1783, mul_881);  view_1783 = mul_881 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_884: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_31, 0.5);  add_31 = None
    mul_885: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_111, view_111)
    mul_886: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_885, -0.5);  mul_885 = None
    exp_47: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_886);  mul_886 = None
    mul_887: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
    mul_888: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_111, mul_887);  view_111 = mul_887 = None
    add_415: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_884, mul_888);  mul_884 = mul_888 = None
    mul_889: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_882, add_415);  mul_882 = add_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1784: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_889, [512, 4096]);  mul_889 = None
    mm_88: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1784, permute_2303);  permute_2303 = None
    permute_2304: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1784, [1, 0])
    mm_89: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_2304, view_110);  permute_2304 = view_110 = None
    permute_2305: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_308: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1784, [0], True);  view_1784 = None
    view_1785: "f32[4096]" = torch.ops.aten.reshape.default(sum_308, [4096]);  sum_308 = None
    permute_2306: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_2305, [1, 0]);  permute_2305 = None
    view_1786: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_88, [512, 1, 1024]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_416: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_877, view_1786);  mul_877 = view_1786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_891: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_416, primals_186);  primals_186 = None
    mul_892: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_891, 1024)
    sum_309: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_891, [2], True)
    mul_893: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_891, mul_21);  mul_891 = None
    sum_310: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_893, [2], True);  mul_893 = None
    mul_894: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_21, sum_310);  sum_310 = None
    sub_226: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_892, sum_309);  mul_892 = sum_309 = None
    sub_227: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_226, mul_894);  sub_226 = mul_894 = None
    mul_895: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_70, sub_227);  div_70 = sub_227 = None
    mul_896: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_416, mul_21);  mul_21 = None
    sum_311: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_896, [0, 1]);  mul_896 = None
    sum_312: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_416, [0, 1]);  add_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_92: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_897: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_92, 1.1111111111111112);  convert_element_type_92 = None
    mul_898: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_895, mul_897);  mul_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1787: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_898, [512, 1, 1024, 1, 1]);  mul_898 = None
    permute_2307: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1787, [0, 3, 4, 1, 2]);  view_1787 = None
    view_1788: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_2307, [1, 512, 1024]);  permute_2307 = None
    bmm_507: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2308, view_1788);  permute_2308 = None
    bmm_508: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1788, permute_2309);  view_1788 = permute_2309 = None
    view_1789: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_507, [64, 16, 1, 1024, 1]);  bmm_507 = None
    permute_2310: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1789, [4, 2, 3, 0, 1]);  view_1789 = None
    view_1790: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_508, [512, 64, 16, 1, 1]);  bmm_508 = None
    permute_2311: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1790, [0, 3, 4, 1, 2]);  view_1790 = None
    permute_2312: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2310, [2, 4, 3, 0, 1]);  permute_2310 = None
    squeeze_484: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2312, 4);  permute_2312 = None
    squeeze_485: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_484, 3);  squeeze_484 = None
    permute_2313: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2311, [0, 1, 4, 3, 2]);  permute_2311 = None
    squeeze_486: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2313, 4);  permute_2313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1791: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_486, [512, 1, 16, 64, 1]);  squeeze_486 = None
    permute_2314: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1791, [2, 0, 4, 1, 3]);  view_1791 = None
    view_1792: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_2314, [16, 512, 64]);  permute_2314 = None
    bmm_509: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_2315, view_1792);  permute_2315 = None
    bmm_510: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1792, permute_2316);  view_1792 = permute_2316 = None
    view_1793: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_509, [16, 512, 1, 64, 1]);  bmm_509 = None
    permute_2317: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1793, [4, 2, 0, 3, 1]);  view_1793 = None
    view_1794: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_510, [16, 512, 512, 1, 1]);  bmm_510 = None
    permute_2318: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1794, [1, 3, 0, 4, 2]);  view_1794 = None
    permute_2319: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2317, [4, 1, 2, 3, 0]);  permute_2317 = None
    squeeze_487: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2319, 4);  permute_2319 = None
    permute_2320: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_2318, [1, 2, 0, 4, 3]);  permute_2318 = None
    squeeze_488: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_2320, 4);  permute_2320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_93: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_29, torch.float32);  getitem_29 = None
    mul_899: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_93, 1.1111111111111112);  convert_element_type_93 = None
    mul_900: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_488, mul_899);  squeeze_488 = mul_899 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_901: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_900, alias_47);  mul_900 = None
    sum_313: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_901, [3], True)
    mul_902: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_47, sum_313);  alias_47 = sum_313 = None
    sub_228: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_901, mul_902);  mul_901 = mul_902 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_903: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_228, 0.125);  sub_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_21: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_903, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1795: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_21, [1, 16, 1023, 512]);  index_put_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_85: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1795, 2, 1, 9223372036854775807);  view_1795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1796: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_85, [1, 16, 512, 1024]);  slice_scatter_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1797: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1796, [1, 16, 512, 1024, 1]);  view_1796 = None
    permute_2321: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1797, [1, 2, 4, 0, 3]);  view_1797 = None
    view_1798: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_2321, [16, 512, 1024]);  permute_2321 = None
    bmm_511: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_2322, view_1798);  permute_2322 = None
    bmm_512: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1798, permute_2323);  view_1798 = permute_2323 = None
    view_1799: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_511, [16, 64, 1, 1024, 1]);  bmm_511 = None
    permute_2324: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1799, [2, 0, 4, 3, 1]);  view_1799 = None
    view_1800: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_512, [16, 512, 64, 1, 1]);  bmm_512 = None
    permute_2325: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1800, [3, 0, 1, 4, 2]);  view_1800 = None
    permute_2326: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2324, [3, 0, 1, 4, 2]);  permute_2324 = None
    squeeze_489: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2326, 4);  permute_2326 = None
    permute_2327: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2325, [2, 0, 1, 4, 3]);  permute_2325 = None
    squeeze_490: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2327, 4);  permute_2327 = None
    sum_314: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_490, [0, 1], True)
    view_1801: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_314, [16, 64]);  sum_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1802: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_903, [1, 16, 512, 512, 1]);  mul_903 = None
    permute_2328: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1802, [1, 2, 4, 0, 3]);  view_1802 = None
    view_1803: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_2328, [16, 512, 512]);  permute_2328 = None
    bmm_513: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_2329, view_1803);  permute_2329 = None
    bmm_514: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1803, permute_2330);  view_1803 = permute_2330 = None
    view_1804: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_513, [16, 64, 1, 512, 1]);  bmm_513 = None
    permute_2331: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1804, [2, 0, 4, 3, 1]);  view_1804 = None
    view_1805: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_514, [16, 512, 64, 1, 1]);  bmm_514 = None
    permute_2332: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1805, [3, 0, 1, 4, 2]);  view_1805 = None
    permute_2333: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2331, [3, 0, 1, 4, 2]);  permute_2331 = None
    squeeze_491: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2333, 4);  permute_2333 = None
    permute_2334: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2332, [2, 0, 1, 4, 3]);  permute_2332 = None
    squeeze_492: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2334, 4);  permute_2334 = None
    sum_315: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_492, [0, 1], True)
    view_1806: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_315, [16, 64]);  sum_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_417: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_490, squeeze_492);  squeeze_490 = squeeze_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1807: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_489, [1024, 1, 16, 64, 1]);  squeeze_489 = None
    permute_2335: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1807, [0, 4, 1, 2, 3]);  view_1807 = None
    view_1808: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_2335, [1, 1024, 1024]);  permute_2335 = None
    bmm_515: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1808);  view_1808 = None
    view_1809: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_515, [1024, 1, 16, 64, 1]);  bmm_515 = None
    permute_2337: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1809, [4, 1, 2, 3, 0]);  view_1809 = None
    permute_2338: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2337, [4, 2, 3, 0, 1]);  permute_2337 = None
    squeeze_493: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2338, 4);  permute_2338 = None
    squeeze_494: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_493, 3);  squeeze_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1810: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_487, [512, 1, 16, 64, 1]);  squeeze_487 = None
    permute_2339: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1810, [0, 4, 1, 2, 3]);  view_1810 = None
    clone_179: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_2339, memory_format = torch.contiguous_format);  permute_2339 = None
    view_1811: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_179, [1, 512, 1024]);  clone_179 = None
    bmm_516: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2340, view_1811)
    bmm_517: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1811, permute_2341);  view_1811 = permute_2341 = None
    view_1812: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_516, [1024, 1, 16, 64, 1]);  bmm_516 = None
    permute_2342: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1812, [4, 1, 2, 3, 0]);  view_1812 = None
    view_1813: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_517, [512, 1024, 1, 1, 1]);  bmm_517 = None
    permute_2343: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1813, [0, 2, 3, 4, 1]);  view_1813 = None
    permute_2344: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2342, [4, 2, 3, 0, 1]);  permute_2342 = None
    squeeze_495: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2344, 4);  permute_2344 = None
    squeeze_496: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_495, 3);  squeeze_495 = None
    permute_2345: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2343, [0, 1, 4, 2, 3]);  permute_2343 = None
    squeeze_497: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2345, 4);  permute_2345 = None
    squeeze_498: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_497, 3);  squeeze_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_418: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_895, squeeze_498);  mul_895 = squeeze_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1814: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_491, [512, 1, 16, 64, 1]);  squeeze_491 = None
    permute_2346: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1814, [0, 4, 1, 2, 3]);  view_1814 = None
    view_1815: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_2346, [1, 512, 1024]);  permute_2346 = None
    bmm_518: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2340, view_1815)
    bmm_519: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1815, permute_2348);  view_1815 = permute_2348 = None
    view_1816: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_518, [1024, 1, 16, 64, 1]);  bmm_518 = None
    permute_2349: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1816, [4, 1, 2, 3, 0]);  view_1816 = None
    view_1817: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_519, [512, 1024, 1, 1, 1]);  bmm_519 = None
    permute_2350: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1817, [0, 2, 3, 4, 1]);  view_1817 = None
    permute_2351: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2349, [4, 2, 3, 0, 1]);  permute_2349 = None
    squeeze_499: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2351, 4);  permute_2351 = None
    squeeze_500: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_499, 3);  squeeze_499 = None
    permute_2352: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2350, [0, 1, 4, 2, 3]);  permute_2350 = None
    squeeze_501: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2352, 4);  permute_2352 = None
    squeeze_502: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_501, 3);  squeeze_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_419: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_418, squeeze_502);  add_418 = squeeze_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1818: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_417, [512, 1, 16, 64, 1]);  add_417 = None
    permute_2353: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1818, [0, 4, 1, 2, 3]);  view_1818 = None
    clone_180: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_2353, memory_format = torch.contiguous_format);  permute_2353 = None
    view_1819: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_180, [1, 512, 1024]);  clone_180 = None
    bmm_520: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2340, view_1819);  permute_2340 = None
    bmm_521: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1819, permute_2355);  view_1819 = permute_2355 = None
    view_1820: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_520, [1024, 1, 16, 64, 1]);  bmm_520 = None
    permute_2356: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1820, [4, 1, 2, 3, 0]);  view_1820 = None
    view_1821: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_521, [512, 1024, 1, 1, 1]);  bmm_521 = None
    permute_2357: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1821, [0, 2, 3, 4, 1]);  view_1821 = None
    permute_2358: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2356, [4, 2, 3, 0, 1]);  permute_2356 = None
    squeeze_503: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2358, 4);  permute_2358 = None
    squeeze_504: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_503, 3);  squeeze_503 = None
    permute_2359: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2357, [0, 1, 4, 2, 3]);  permute_2357 = None
    squeeze_505: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2359, 4);  permute_2359 = None
    squeeze_506: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_505, 3);  squeeze_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_420: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_419, squeeze_506);  add_419 = squeeze_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_905: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_420, primals_184);  primals_184 = None
    mul_906: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_905, 1024)
    sum_316: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_905, [2], True)
    mul_907: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_905, mul_18);  mul_905 = None
    sum_317: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_907, [2], True);  mul_907 = None
    mul_908: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_18, sum_317);  sum_317 = None
    sub_230: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_906, sum_316);  mul_906 = sum_316 = None
    sub_231: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_230, mul_908);  sub_230 = mul_908 = None
    mul_909: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_71, sub_231);  div_71 = sub_231 = None
    mul_910: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_420, mul_18);  mul_18 = None
    sum_318: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_910, [0, 1]);  mul_910 = None
    sum_319: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_420, [0, 1]);  add_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_94: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_25, torch.float32);  getitem_25 = None
    mul_911: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_94, 1.1111111111111112);  convert_element_type_94 = None
    mul_912: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_909, mul_911);  mul_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1822: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_912, [512, 1024]);  mul_912 = None
    mm_90: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1822, permute_2360);  permute_2360 = None
    permute_2361: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1822, [1, 0])
    mm_91: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_2361, view_74);  permute_2361 = view_74 = None
    permute_2362: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_320: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1822, [0], True);  view_1822 = None
    view_1823: "f32[1024]" = torch.ops.aten.reshape.default(sum_320, [1024]);  sum_320 = None
    permute_2363: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_2362, [1, 0]);  permute_2362 = None
    view_1824: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_90, [512, 1, 4096]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_95: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_23, torch.float32);  getitem_23 = None
    mul_913: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_95, 1.1111111111111112);  convert_element_type_95 = None
    mul_914: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1824, mul_913);  view_1824 = mul_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_916: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_20, 0.5);  add_20 = None
    mul_917: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_73, view_73)
    mul_918: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_917, -0.5);  mul_917 = None
    exp_48: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_918);  mul_918 = None
    mul_919: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_48, 0.3989422804014327);  exp_48 = None
    mul_920: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_73, mul_919);  view_73 = mul_919 = None
    add_422: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_916, mul_920);  mul_916 = mul_920 = None
    mul_921: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_914, add_422);  mul_914 = add_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1825: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_921, [512, 4096]);  mul_921 = None
    mm_92: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1825, permute_2364);  permute_2364 = None
    permute_2365: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1825, [1, 0])
    mm_93: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_2365, view_72);  permute_2365 = view_72 = None
    permute_2366: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_321: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1825, [0], True);  view_1825 = None
    view_1826: "f32[4096]" = torch.ops.aten.reshape.default(sum_321, [4096]);  sum_321 = None
    permute_2367: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_2366, [1, 0]);  permute_2366 = None
    view_1827: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_92, [512, 1, 1024]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_423: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_909, view_1827);  mul_909 = view_1827 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_923: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_423, primals_178);  primals_178 = None
    mul_924: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_923, 1024)
    sum_322: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_923, [2], True)
    mul_925: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_923, mul_13);  mul_923 = None
    sum_323: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_925, [2], True);  mul_925 = None
    mul_926: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_13, sum_323);  sum_323 = None
    sub_233: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_924, sum_322);  mul_924 = sum_322 = None
    sub_234: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_233, mul_926);  sub_233 = mul_926 = None
    mul_927: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_72, sub_234);  div_72 = sub_234 = None
    mul_928: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_423, mul_13);  mul_13 = None
    sum_324: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_928, [0, 1]);  mul_928 = None
    sum_325: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_423, [0, 1]);  add_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_96: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_19, torch.float32);  getitem_19 = None
    mul_929: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_96, 1.1111111111111112);  convert_element_type_96 = None
    mul_930: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_927, mul_929);  mul_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1828: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_930, [512, 1, 1024, 1, 1]);  mul_930 = None
    permute_2368: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1828, [0, 3, 4, 1, 2]);  view_1828 = None
    view_1829: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_2368, [1, 512, 1024]);  permute_2368 = None
    bmm_522: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2369, view_1829);  permute_2369 = None
    bmm_523: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1829, permute_2370);  view_1829 = permute_2370 = None
    view_1830: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_522, [64, 16, 1, 1024, 1]);  bmm_522 = None
    permute_2371: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1830, [4, 2, 3, 0, 1]);  view_1830 = None
    view_1831: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_523, [512, 64, 16, 1, 1]);  bmm_523 = None
    permute_2372: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1831, [0, 3, 4, 1, 2]);  view_1831 = None
    permute_2373: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2371, [2, 4, 3, 0, 1]);  permute_2371 = None
    squeeze_507: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2373, 4);  permute_2373 = None
    squeeze_508: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_507, 3);  squeeze_507 = None
    permute_2374: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2372, [0, 1, 4, 3, 2]);  permute_2372 = None
    squeeze_509: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2374, 4);  permute_2374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1832: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_509, [512, 1, 16, 64, 1]);  squeeze_509 = None
    permute_2375: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1832, [2, 0, 4, 1, 3]);  view_1832 = None
    view_1833: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_2375, [16, 512, 64]);  permute_2375 = None
    bmm_524: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_2376, view_1833);  permute_2376 = None
    bmm_525: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1833, permute_2377);  view_1833 = permute_2377 = None
    view_1834: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_524, [16, 512, 1, 64, 1]);  bmm_524 = None
    permute_2378: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1834, [4, 2, 0, 3, 1]);  view_1834 = None
    view_1835: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_525, [16, 512, 512, 1, 1]);  bmm_525 = None
    permute_2379: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1835, [1, 3, 0, 4, 2]);  view_1835 = None
    permute_2380: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2378, [4, 1, 2, 3, 0]);  permute_2378 = None
    squeeze_510: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2380, 4);  permute_2380 = None
    permute_2381: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_2379, [1, 2, 0, 4, 3]);  permute_2379 = None
    squeeze_511: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_2381, 4);  permute_2381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_97: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_931: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_97, 1.1111111111111112);  convert_element_type_97 = None
    mul_932: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_511, mul_931);  squeeze_511 = mul_931 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_933: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_932, alias_48);  mul_932 = None
    sum_326: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_933, [3], True)
    mul_934: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_48, sum_326);  alias_48 = sum_326 = None
    sub_235: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_933, mul_934);  mul_933 = mul_934 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_935: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_235, 0.125);  sub_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_22: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default_5, [None, None, None, iota], mul_935, True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1836: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_22, [1, 16, 1023, 512]);  index_put_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_89: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1836, 2, 1, 9223372036854775807);  view_1836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1837: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_89, [1, 16, 512, 1024]);  slice_scatter_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1838: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1837, [1, 16, 512, 1024, 1]);  view_1837 = None
    permute_2382: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1838, [1, 2, 4, 0, 3]);  view_1838 = None
    view_1839: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_2382, [16, 512, 1024]);  permute_2382 = None
    bmm_526: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_2383, view_1839);  permute_2383 = None
    bmm_527: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1839, permute_2384);  view_1839 = permute_2384 = None
    view_1840: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_526, [16, 64, 1, 1024, 1]);  bmm_526 = None
    permute_2385: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1840, [2, 0, 4, 3, 1]);  view_1840 = None
    view_1841: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_527, [16, 512, 64, 1, 1]);  bmm_527 = None
    permute_2386: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1841, [3, 0, 1, 4, 2]);  view_1841 = None
    permute_2387: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2385, [3, 0, 1, 4, 2]);  permute_2385 = None
    squeeze_512: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2387, 4);  permute_2387 = None
    permute_2388: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2386, [2, 0, 1, 4, 3]);  permute_2386 = None
    squeeze_513: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2388, 4);  permute_2388 = None
    sum_327: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_513, [0, 1], True)
    view_1842: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_327, [16, 64]);  sum_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1843: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_935, [1, 16, 512, 512, 1]);  mul_935 = None
    permute_2389: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1843, [1, 2, 4, 0, 3]);  view_1843 = None
    view_1844: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_2389, [16, 512, 512]);  permute_2389 = None
    bmm_528: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_2390, view_1844);  permute_2390 = None
    bmm_529: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1844, permute_2391);  view_1844 = permute_2391 = None
    view_1845: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_528, [16, 64, 1, 512, 1]);  bmm_528 = None
    permute_2392: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1845, [2, 0, 4, 3, 1]);  view_1845 = None
    view_1846: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_529, [16, 512, 64, 1, 1]);  bmm_529 = None
    permute_2393: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1846, [3, 0, 1, 4, 2]);  view_1846 = None
    permute_2394: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2392, [3, 0, 1, 4, 2]);  permute_2392 = None
    squeeze_514: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2394, 4);  permute_2394 = None
    permute_2395: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2393, [2, 0, 1, 4, 3]);  permute_2393 = None
    squeeze_515: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2395, 4);  permute_2395 = None
    sum_328: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_515, [0, 1], True)
    view_1847: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_328, [16, 64]);  sum_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_424: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_513, squeeze_515);  squeeze_513 = squeeze_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1848: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_512, [1024, 1, 16, 64, 1]);  squeeze_512 = None
    permute_2396: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1848, [0, 4, 1, 2, 3]);  view_1848 = None
    view_1849: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_2396, [1, 1024, 1024]);  permute_2396 = None
    bmm_530: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1849);  view_1849 = None
    view_1850: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_530, [1024, 1, 16, 64, 1]);  bmm_530 = None
    permute_2398: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1850, [4, 1, 2, 3, 0]);  view_1850 = None
    permute_2399: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2398, [4, 2, 3, 0, 1]);  permute_2398 = None
    squeeze_516: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2399, 4);  permute_2399 = None
    squeeze_517: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_516, 3);  squeeze_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1851: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_510, [512, 1, 16, 64, 1]);  squeeze_510 = None
    permute_2400: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1851, [0, 4, 1, 2, 3]);  view_1851 = None
    clone_185: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_2400, memory_format = torch.contiguous_format);  permute_2400 = None
    view_1852: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_185, [1, 512, 1024]);  clone_185 = None
    bmm_531: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2401, view_1852)
    bmm_532: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1852, permute_2402);  view_1852 = permute_2402 = None
    view_1853: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_531, [1024, 1, 16, 64, 1]);  bmm_531 = None
    permute_2403: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1853, [4, 1, 2, 3, 0]);  view_1853 = None
    view_1854: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_532, [512, 1024, 1, 1, 1]);  bmm_532 = None
    permute_2404: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1854, [0, 2, 3, 4, 1]);  view_1854 = None
    permute_2405: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2403, [4, 2, 3, 0, 1]);  permute_2403 = None
    squeeze_518: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2405, 4);  permute_2405 = None
    squeeze_519: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_518, 3);  squeeze_518 = None
    permute_2406: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2404, [0, 1, 4, 2, 3]);  permute_2404 = None
    squeeze_520: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2406, 4);  permute_2406 = None
    squeeze_521: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_520, 3);  squeeze_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_425: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_927, squeeze_521);  mul_927 = squeeze_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1855: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_514, [512, 1, 16, 64, 1]);  squeeze_514 = None
    permute_2407: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1855, [0, 4, 1, 2, 3]);  view_1855 = None
    view_1856: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_2407, [1, 512, 1024]);  permute_2407 = None
    bmm_533: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2401, view_1856)
    bmm_534: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1856, permute_2409);  view_1856 = permute_2409 = None
    view_1857: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_533, [1024, 1, 16, 64, 1]);  bmm_533 = None
    permute_2410: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1857, [4, 1, 2, 3, 0]);  view_1857 = None
    view_1858: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_534, [512, 1024, 1, 1, 1]);  bmm_534 = None
    permute_2411: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1858, [0, 2, 3, 4, 1]);  view_1858 = None
    permute_2412: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2410, [4, 2, 3, 0, 1]);  permute_2410 = None
    squeeze_522: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2412, 4);  permute_2412 = None
    squeeze_523: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_522, 3);  squeeze_522 = None
    permute_2413: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2411, [0, 1, 4, 2, 3]);  permute_2411 = None
    squeeze_524: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2413, 4);  permute_2413 = None
    squeeze_525: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_524, 3);  squeeze_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_426: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_425, squeeze_525);  add_425 = squeeze_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1859: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_424, [512, 1, 16, 64, 1]);  add_424 = None
    permute_2414: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1859, [0, 4, 1, 2, 3]);  view_1859 = None
    clone_186: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_2414, memory_format = torch.contiguous_format);  permute_2414 = None
    view_1860: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_186, [1, 512, 1024]);  clone_186 = None
    bmm_535: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2401, view_1860);  permute_2401 = None
    bmm_536: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1860, permute_2416);  view_1860 = permute_2416 = None
    view_1861: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_535, [1024, 1, 16, 64, 1]);  bmm_535 = None
    permute_2417: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1861, [4, 1, 2, 3, 0]);  view_1861 = None
    view_1862: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_536, [512, 1024, 1, 1, 1]);  bmm_536 = None
    permute_2418: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1862, [0, 2, 3, 4, 1]);  view_1862 = None
    permute_2419: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2417, [4, 2, 3, 0, 1]);  permute_2417 = None
    squeeze_526: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2419, 4);  permute_2419 = None
    squeeze_527: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_526, 3);  squeeze_526 = None
    permute_2420: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2418, [0, 1, 4, 2, 3]);  permute_2418 = None
    squeeze_528: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2420, 4);  permute_2420 = None
    squeeze_529: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_528, 3);  squeeze_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_427: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_426, squeeze_529);  add_426 = squeeze_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    mul_937: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_427, primals_176);  primals_176 = None
    mul_938: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_937, 1024)
    sum_329: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_937, [2], True)
    mul_939: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_937, mul_10);  mul_937 = None
    sum_330: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_939, [2], True);  mul_939 = None
    mul_940: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_10, sum_330);  sum_330 = None
    sub_237: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_938, sum_329);  mul_938 = sum_329 = None
    sub_238: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_237, mul_940);  sub_237 = mul_940 = None
    mul_941: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_73, sub_238);  div_73 = sub_238 = None
    mul_942: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_427, mul_10);  mul_10 = None
    sum_331: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_942, [0, 1]);  mul_942 = None
    sum_332: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_427, [0, 1]);  add_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    convert_element_type_98: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
    mul_943: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_98, 1.1111111111111112);  convert_element_type_98 = None
    mul_944: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_941, mul_943);  mul_943 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    view_1863: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_944, [512, 1024]);  mul_944 = None
    mm_94: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1863, permute_2421);  permute_2421 = None
    permute_2422: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1863, [1, 0])
    mm_95: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_2422, view_36);  permute_2422 = view_36 = None
    permute_2423: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_333: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1863, [0], True);  view_1863 = None
    view_1864: "f32[1024]" = torch.ops.aten.reshape.default(sum_333, [1024]);  sum_333 = None
    permute_2424: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_2423, [1, 0]);  permute_2423 = None
    view_1865: "f32[512, 1, 4096]" = torch.ops.aten.reshape.default(mm_94, [512, 1, 4096]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    convert_element_type_99: "f32[512, 1, 4096]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_945: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_99, 1.1111111111111112);  convert_element_type_99 = None
    mul_946: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_1865, mul_945);  view_1865 = mul_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_948: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
    mul_949: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_35, view_35)
    mul_950: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_949, -0.5);  mul_949 = None
    exp_49: "f32[512, 1, 4096]" = torch.ops.aten.exp.default(mul_950);  mul_950 = None
    mul_951: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(exp_49, 0.3989422804014327);  exp_49 = None
    mul_952: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(view_35, mul_951);  view_35 = mul_951 = None
    add_429: "f32[512, 1, 4096]" = torch.ops.aten.add.Tensor(mul_948, mul_952);  mul_948 = mul_952 = None
    mul_953: "f32[512, 1, 4096]" = torch.ops.aten.mul.Tensor(mul_946, add_429);  mul_946 = add_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    view_1866: "f32[512, 4096]" = torch.ops.aten.reshape.default(mul_953, [512, 4096]);  mul_953 = None
    mm_96: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1866, permute_2425);  permute_2425 = None
    permute_2426: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1866, [1, 0])
    mm_97: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_2426, view_34);  permute_2426 = view_34 = None
    permute_2427: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_334: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1866, [0], True);  view_1866 = None
    view_1867: "f32[4096]" = torch.ops.aten.reshape.default(sum_334, [4096]);  sum_334 = None
    permute_2428: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_2427, [1, 0]);  permute_2427 = None
    view_1868: "f32[512, 1, 1024]" = torch.ops.aten.reshape.default(mm_96, [512, 1, 1024]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    add_430: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_941, view_1868);  mul_941 = view_1868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    mul_955: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_430, primals_170);  primals_170 = None
    mul_956: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_955, 1024)
    sum_335: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_955, [2], True)
    mul_957: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_955, mul_5);  mul_955 = None
    sum_336: "f32[512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_957, [2], True);  mul_957 = None
    mul_958: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_5, sum_336);  sum_336 = None
    sub_240: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_956, sum_335);  mul_956 = sum_335 = None
    sub_241: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_240, mul_958);  sub_240 = mul_958 = None
    mul_959: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(div_74, sub_241);  div_74 = sub_241 = None
    mul_960: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_430, mul_5);  mul_5 = None
    sum_337: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_960, [0, 1]);  mul_960 = None
    sum_338: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_430, [0, 1]);  add_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    convert_element_type_100: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_961: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_100, 1.1111111111111112);  convert_element_type_100 = None
    mul_962: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_959, mul_961);  mul_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    view_1869: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.reshape.default(mul_962, [512, 1, 1024, 1, 1]);  mul_962 = None
    permute_2429: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1869, [0, 3, 4, 1, 2]);  view_1869 = None
    view_1870: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_2429, [1, 512, 1024]);  permute_2429 = None
    bmm_537: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2430, view_1870);  permute_2430 = None
    bmm_538: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1870, permute_2431);  view_1870 = permute_2431 = None
    view_1871: "f32[64, 16, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_537, [64, 16, 1, 1024, 1]);  bmm_537 = None
    permute_2432: "f32[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_1871, [4, 2, 3, 0, 1]);  view_1871 = None
    view_1872: "f32[512, 64, 16, 1, 1]" = torch.ops.aten.reshape.default(bmm_538, [512, 64, 16, 1, 1]);  bmm_538 = None
    permute_2433: "f32[512, 1, 1, 64, 16]" = torch.ops.aten.permute.default(view_1872, [0, 3, 4, 1, 2]);  view_1872 = None
    permute_2434: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2432, [2, 4, 3, 0, 1]);  permute_2432 = None
    squeeze_530: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2434, 4);  permute_2434 = None
    squeeze_531: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_530, 3);  squeeze_530 = None
    permute_2435: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2433, [0, 1, 4, 3, 2]);  permute_2433 = None
    squeeze_532: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2435, 4);  permute_2435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    view_1873: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_532, [512, 1, 16, 64, 1]);  squeeze_532 = None
    permute_2436: "f32[16, 512, 1, 1, 64]" = torch.ops.aten.permute.default(view_1873, [2, 0, 4, 1, 3]);  view_1873 = None
    view_1874: "f32[16, 512, 64]" = torch.ops.aten.reshape.default(permute_2436, [16, 512, 64]);  permute_2436 = None
    bmm_539: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_2437, view_1874);  permute_2437 = None
    bmm_540: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_1874, permute_2438);  view_1874 = permute_2438 = None
    view_1875: "f32[16, 512, 1, 64, 1]" = torch.ops.aten.reshape.default(bmm_539, [16, 512, 1, 64, 1]);  bmm_539 = None
    permute_2439: "f32[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(view_1875, [4, 2, 0, 3, 1]);  view_1875 = None
    view_1876: "f32[16, 512, 512, 1, 1]" = torch.ops.aten.reshape.default(bmm_540, [16, 512, 512, 1, 1]);  bmm_540 = None
    permute_2440: "f32[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(view_1876, [1, 3, 0, 4, 2]);  view_1876 = None
    permute_2441: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2439, [4, 1, 2, 3, 0]);  permute_2439 = None
    squeeze_533: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2441, 4);  permute_2441 = None
    permute_2442: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(permute_2440, [1, 2, 0, 4, 3]);  permute_2440 = None
    squeeze_534: "f32[1, 16, 512, 512]" = torch.ops.aten.squeeze.dim(permute_2442, 4);  permute_2442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    convert_element_type_101: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.float32);  getitem_5 = None
    mul_963: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_101, 1.1111111111111112);  convert_element_type_101 = None
    mul_964: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_534, mul_963);  squeeze_534 = mul_963 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    mul_965: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_964, alias_49);  mul_964 = None
    sum_339: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_965, [3], True)
    mul_966: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_49, sum_339);  alias_49 = sum_339 = None
    sub_242: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_965, mul_966);  mul_965 = mul_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    mul_967: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_242, 0.125);  sub_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    index_put_23: "f32[1, 16, 512, 1023]" = torch.ops.aten.index_put_.default(full_default_5, [None, None, None, iota], mul_967, True);  full_default_5 = iota = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    view_1877: "f32[1, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_23, [1, 16, 1023, 512]);  index_put_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    slice_scatter_93: "f32[1, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_7, view_1877, 2, 1, 9223372036854775807);  full_default_7 = view_1877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    view_1878: "f32[1, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_93, [1, 16, 512, 1024]);  slice_scatter_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    view_1879: "f32[1, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(view_1878, [1, 16, 512, 1024, 1]);  view_1878 = None
    permute_2443: "f32[16, 512, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1879, [1, 2, 4, 0, 3]);  view_1879 = None
    view_1880: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_2443, [16, 512, 1024]);  permute_2443 = None
    bmm_541: "f32[16, 64, 1024]" = torch.ops.aten.bmm.default(permute_2444, view_1880);  permute_2444 = None
    bmm_542: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1880, permute_2445);  view_1880 = permute_2445 = None
    view_1881: "f32[16, 64, 1, 1024, 1]" = torch.ops.aten.reshape.default(bmm_541, [16, 64, 1, 1024, 1]);  bmm_541 = None
    permute_2446: "f32[1, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_1881, [2, 0, 4, 3, 1]);  view_1881 = None
    view_1882: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_542, [16, 512, 64, 1, 1]);  bmm_542 = None
    permute_2447: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1882, [3, 0, 1, 4, 2]);  view_1882 = None
    permute_2448: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2446, [3, 0, 1, 4, 2]);  permute_2446 = None
    squeeze_535: "f32[1024, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2448, 4);  permute_2448 = None
    permute_2449: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2447, [2, 0, 1, 4, 3]);  permute_2447 = None
    squeeze_536: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2449, 4);  permute_2449 = None
    sum_340: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_536, [0, 1], True)
    view_1883: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_340, [16, 64]);  sum_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    view_1884: "f32[1, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_967, [1, 16, 512, 512, 1]);  mul_967 = None
    permute_2450: "f32[16, 512, 1, 1, 512]" = torch.ops.aten.permute.default(view_1884, [1, 2, 4, 0, 3]);  view_1884 = None
    view_1885: "f32[16, 512, 512]" = torch.ops.aten.reshape.default(permute_2450, [16, 512, 512]);  permute_2450 = None
    bmm_543: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_2451, view_1885);  permute_2451 = None
    bmm_544: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_1885, permute_2452);  view_1885 = permute_2452 = None
    view_1886: "f32[16, 64, 1, 512, 1]" = torch.ops.aten.reshape.default(bmm_543, [16, 64, 1, 512, 1]);  bmm_543 = None
    permute_2453: "f32[1, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_1886, [2, 0, 4, 3, 1]);  view_1886 = None
    view_1887: "f32[16, 512, 64, 1, 1]" = torch.ops.aten.reshape.default(bmm_544, [16, 512, 64, 1, 1]);  bmm_544 = None
    permute_2454: "f32[1, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_1887, [3, 0, 1, 4, 2]);  view_1887 = None
    permute_2455: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2453, [3, 0, 1, 4, 2]);  permute_2453 = None
    squeeze_537: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2455, 4);  permute_2455 = None
    permute_2456: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2454, [2, 0, 1, 4, 3]);  permute_2454 = None
    squeeze_538: "f32[512, 1, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2456, 4);  permute_2456 = None
    sum_341: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(squeeze_538, [0, 1], True)
    view_1888: "f32[16, 64]" = torch.ops.aten.reshape.default(sum_341, [16, 64]);  sum_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    add_431: "f32[512, 1, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_536, squeeze_538);  squeeze_536 = squeeze_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    view_1889: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_535, [1024, 1, 16, 64, 1]);  squeeze_535 = None
    permute_2457: "f32[1024, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1889, [0, 4, 1, 2, 3]);  view_1889 = None
    view_1890: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(permute_2457, [1, 1024, 1024]);  permute_2457 = None
    bmm_545: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_1055, view_1890);  permute_1055 = view_1890 = None
    view_1891: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_545, [1024, 1, 16, 64, 1]);  bmm_545 = None
    permute_2459: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1891, [4, 1, 2, 3, 0]);  view_1891 = None
    permute_2460: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2459, [4, 2, 3, 0, 1]);  permute_2459 = None
    squeeze_539: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2460, 4);  permute_2460 = None
    squeeze_540: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_539, 3);  squeeze_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    view_1892: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_533, [512, 1, 16, 64, 1]);  squeeze_533 = None
    permute_2461: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1892, [0, 4, 1, 2, 3]);  view_1892 = None
    clone_191: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_2461, memory_format = torch.contiguous_format);  permute_2461 = None
    view_1893: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_191, [1, 512, 1024]);  clone_191 = None
    bmm_546: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2462, view_1893)
    bmm_547: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1893, permute_2463);  view_1893 = permute_2463 = None
    view_1894: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_546, [1024, 1, 16, 64, 1]);  bmm_546 = None
    permute_2464: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1894, [4, 1, 2, 3, 0]);  view_1894 = None
    view_1895: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_547, [512, 1024, 1, 1, 1]);  bmm_547 = None
    permute_2465: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1895, [0, 2, 3, 4, 1]);  view_1895 = None
    permute_2466: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2464, [4, 2, 3, 0, 1]);  permute_2464 = None
    squeeze_541: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2466, 4);  permute_2466 = None
    squeeze_542: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_541, 3);  squeeze_541 = None
    permute_2467: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2465, [0, 1, 4, 2, 3]);  permute_2465 = None
    squeeze_543: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2467, 4);  permute_2467 = None
    squeeze_544: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_543, 3);  squeeze_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    add_432: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_959, squeeze_544);  mul_959 = squeeze_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    view_1896: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_537, [512, 1, 16, 64, 1]);  squeeze_537 = None
    permute_2468: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1896, [0, 4, 1, 2, 3]);  view_1896 = None
    view_1897: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(permute_2468, [1, 512, 1024]);  permute_2468 = None
    bmm_548: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2462, view_1897)
    bmm_549: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1897, permute_2470);  view_1897 = permute_2470 = None
    view_1898: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_548, [1024, 1, 16, 64, 1]);  bmm_548 = None
    permute_2471: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1898, [4, 1, 2, 3, 0]);  view_1898 = None
    view_1899: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_549, [512, 1024, 1, 1, 1]);  bmm_549 = None
    permute_2472: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1899, [0, 2, 3, 4, 1]);  view_1899 = None
    permute_2473: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2471, [4, 2, 3, 0, 1]);  permute_2471 = None
    squeeze_545: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2473, 4);  permute_2473 = None
    squeeze_546: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_545, 3);  squeeze_545 = None
    permute_2474: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2472, [0, 1, 4, 2, 3]);  permute_2472 = None
    squeeze_547: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2474, 4);  permute_2474 = None
    squeeze_548: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_547, 3);  squeeze_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    add_433: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_432, squeeze_548);  add_432 = squeeze_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    view_1900: "f32[512, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(add_431, [512, 1, 16, 64, 1]);  add_431 = None
    permute_2475: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.permute.default(view_1900, [0, 4, 1, 2, 3]);  view_1900 = None
    clone_192: "f32[512, 1, 1, 16, 64]" = torch.ops.aten.clone.default(permute_2475, memory_format = torch.contiguous_format);  permute_2475 = None
    view_1901: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(clone_192, [1, 512, 1024]);  clone_192 = None
    bmm_550: "f32[1, 1024, 1024]" = torch.ops.aten.bmm.default(permute_2462, view_1901);  permute_2462 = None
    bmm_551: "f32[1, 512, 1024]" = torch.ops.aten.bmm.default(view_1901, permute_2477);  view_1901 = permute_2477 = None
    view_1902: "f32[1024, 1, 16, 64, 1]" = torch.ops.aten.reshape.default(bmm_550, [1024, 1, 16, 64, 1]);  bmm_550 = None
    permute_2478: "f32[1, 1, 16, 64, 1024]" = torch.ops.aten.permute.default(view_1902, [4, 1, 2, 3, 0]);  view_1902 = None
    view_1903: "f32[512, 1024, 1, 1, 1]" = torch.ops.aten.reshape.default(bmm_551, [512, 1024, 1, 1, 1]);  bmm_551 = None
    permute_2479: "f32[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(view_1903, [0, 2, 3, 4, 1]);  view_1903 = None
    permute_2480: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_2478, [4, 2, 3, 0, 1]);  permute_2478 = None
    squeeze_549: "f32[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_2480, 4);  permute_2480 = None
    squeeze_550: "f32[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_549, 3);  squeeze_549 = None
    permute_2481: "f32[512, 1, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_2479, [0, 1, 4, 2, 3]);  permute_2479 = None
    squeeze_551: "f32[512, 1, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_2481, 4);  permute_2481 = None
    squeeze_552: "f32[512, 1, 1024]" = torch.ops.aten.squeeze.dim(squeeze_551, 3);  squeeze_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    add_434: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(add_433, squeeze_552);  add_433 = squeeze_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1177, code: output_h = self.dropout(word_emb_k)
    convert_element_type_102: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
    mul_968: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_102, 1.1111111111111112);  convert_element_type_102 = None
    mul_969: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(add_434, mul_968);  add_434 = mul_968 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1176, code: word_emb_k = self.word_embedding(input_ids)
    eq: "b8[512, 1]" = torch.ops.aten.eq.Scalar(permute, -1)
    unsqueeze_605: "b8[512, 1, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    where_4: "f32[512, 1, 1024]" = torch.ops.aten.where.self(unsqueeze_605, full_default_1, mul_969);  unsqueeze_605 = full_default_1 = mul_969 = None
    full_default_126: "f32[32000, 1024]" = torch.ops.aten.full.default([32000, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[32000, 1024]" = torch.ops.prims._unsafe_index_put_.default(full_default_126, [permute], where_4, True);  full_default_126 = permute = where_4 = None
    return [squeeze_550, squeeze_546, squeeze_542, squeeze_540, view_1888, view_1883, squeeze_531, squeeze_527, squeeze_523, squeeze_519, squeeze_517, view_1847, view_1842, squeeze_508, squeeze_504, squeeze_500, squeeze_496, squeeze_494, view_1806, view_1801, squeeze_485, squeeze_481, squeeze_477, squeeze_473, squeeze_471, view_1765, view_1760, squeeze_462, squeeze_458, squeeze_454, squeeze_450, squeeze_448, view_1724, view_1719, squeeze_439, squeeze_435, squeeze_431, squeeze_427, squeeze_425, view_1683, view_1678, squeeze_416, squeeze_412, squeeze_408, squeeze_404, squeeze_402, view_1642, view_1637, squeeze_393, squeeze_389, squeeze_385, squeeze_381, squeeze_379, view_1601, view_1596, squeeze_370, squeeze_366, squeeze_362, squeeze_358, squeeze_356, view_1560, view_1555, squeeze_347, squeeze_343, squeeze_339, squeeze_335, squeeze_333, view_1519, view_1514, squeeze_324, squeeze_320, squeeze_316, squeeze_312, squeeze_310, view_1478, view_1473, squeeze_301, squeeze_297, squeeze_293, squeeze_289, squeeze_287, view_1437, view_1432, squeeze_278, squeeze_274, squeeze_270, squeeze_266, squeeze_264, view_1396, view_1391, squeeze_255, squeeze_251, squeeze_247, squeeze_243, squeeze_241, view_1355, view_1350, squeeze_232, squeeze_228, squeeze_224, squeeze_220, squeeze_218, view_1314, view_1309, squeeze_209, squeeze_205, squeeze_201, squeeze_197, squeeze_195, view_1273, view_1268, squeeze_186, squeeze_182, squeeze_178, squeeze_174, squeeze_172, view_1232, view_1227, squeeze_163, squeeze_159, squeeze_155, squeeze_151, squeeze_149, view_1191, view_1186, squeeze_140, squeeze_136, squeeze_132, squeeze_128, squeeze_126, view_1150, view_1145, squeeze_117, squeeze_113, squeeze_109, squeeze_105, squeeze_103, view_1109, view_1104, squeeze_94, squeeze_90, squeeze_86, squeeze_82, squeeze_80, view_1068, view_1063, squeeze_71, squeeze_67, squeeze_63, squeeze_59, squeeze_57, view_1027, view_1022, squeeze_48, squeeze_44, squeeze_40, squeeze_36, squeeze_34, view_986, view_981, squeeze_25, squeeze_21, squeeze_17, squeeze_13, squeeze_11, view_945, view_940, squeeze_2, _unsafe_index_put, sum_337, sum_338, permute_2428, view_1867, permute_2424, view_1864, sum_331, sum_332, sum_324, sum_325, permute_2367, view_1826, permute_2363, view_1823, sum_318, sum_319, sum_311, sum_312, permute_2306, view_1785, permute_2302, view_1782, sum_305, sum_306, sum_298, sum_299, permute_2245, view_1744, permute_2241, view_1741, sum_292, sum_293, sum_285, sum_286, permute_2184, view_1703, permute_2180, view_1700, sum_279, sum_280, sum_272, sum_273, permute_2123, view_1662, permute_2119, view_1659, sum_266, sum_267, sum_259, sum_260, permute_2062, view_1621, permute_2058, view_1618, sum_253, sum_254, sum_246, sum_247, permute_2001, view_1580, permute_1997, view_1577, sum_240, sum_241, sum_233, sum_234, permute_1940, view_1539, permute_1936, view_1536, sum_227, sum_228, sum_220, sum_221, permute_1879, view_1498, permute_1875, view_1495, sum_214, sum_215, sum_207, sum_208, permute_1818, view_1457, permute_1814, view_1454, sum_201, sum_202, sum_194, sum_195, permute_1757, view_1416, permute_1753, view_1413, sum_188, sum_189, sum_181, sum_182, permute_1696, view_1375, permute_1692, view_1372, sum_175, sum_176, sum_168, sum_169, permute_1635, view_1334, permute_1631, view_1331, sum_162, sum_163, sum_155, sum_156, permute_1574, view_1293, permute_1570, view_1290, sum_149, sum_150, sum_142, sum_143, permute_1513, view_1252, permute_1509, view_1249, sum_136, sum_137, sum_129, sum_130, permute_1452, view_1211, permute_1448, view_1208, sum_123, sum_124, sum_116, sum_117, permute_1391, view_1170, permute_1387, view_1167, sum_110, sum_111, sum_103, sum_104, permute_1330, view_1129, permute_1326, view_1126, sum_97, sum_98, sum_90, sum_91, permute_1269, view_1088, permute_1265, view_1085, sum_84, sum_85, sum_77, sum_78, permute_1208, view_1047, permute_1204, view_1044, sum_71, sum_72, sum_64, sum_65, permute_1147, view_1006, permute_1143, view_1003, sum_58, sum_59, sum_51, sum_52, permute_1086, view_965, permute_1082, view_962, sum_45, sum_46, sum_38, sum_39, permute_1025, view_924, permute_1021, view_921, sum_32, sum_33, permute_1016, view_918, None, None]
    