/*
 Navicat Premium Data Transfer

 Source Server         : elec5619
 Source Server Type    : MySQL
 Source Server Version : 80019
 Source Host           : localhost:3306
 Source Schema         : Fitastic

 Target Server Type    : MySQL
 Target Server Version : 80019
 File Encoding         : 65001

 Date: 27/04/2021 19:03:40
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for users
-- ----------------------------
DROP TABLE IF EXISTS `users`;
CREATE TABLE `users` (
  `id` int NOT NULL,
  `user_id` int NOT NULL,
  `username` varchar(100) DEFAULT NULL,
  `password` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Records of users
-- ----------------------------
BEGIN;
INSERT INTO `users` VALUES (1, 11111116, 'Admin', '123456');
INSERT INTO `users` VALUES (2, 222, 'Eason', '888888');
INSERT INTO `users` VALUES (3, 333, 'Tommy', '666666');
COMMIT;

SET FOREIGN_KEY_CHECKS = 1;
